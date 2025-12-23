import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


# ==============================
# 1) Inverted Pendulum Environment
# ==============================
class InvertedPendulumEnv(gym.Env):
    """
    Custom inverted pendulum environment (cart + pole, linearized).
    State  : [x, x_dot, theta, theta_dot]
    Action : horizontal force on the cart (continuous)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.02,
        max_episode_steps: int = 1500,  # 30 s with dt=0.02
        seed: int | None = None,
    ):
        super().__init__()

        # Discrete-time step
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # Linearized system matrices (around upright)
        # dx/dt = A x + B u
        self.A = np.array(
            [[0, 1, 0, 0], [0, 0, -0.7171, 0], [0, 0, 0, 1], [0, 0, 31.5512, 0]],
            dtype=np.float64,
        )
        self.B = np.array([0, 0.9756, 0, -2.9268], dtype=np.float64)

        # Physical limits
        self.x_threshold = 2.4  # m
        self.theta_threshold_radians = 12 * np.pi / 180.0  # rad

        # Observation space
        high = np.array(
            [
                self.x_threshold * 1.5,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 1.5,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Action space (continuous force)
        self.action_space = spaces.Box(
            low=np.array([-15.0], dtype=np.float32),
            high=np.array([15.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Scales used in reward shaping for velocities
        self.xdot_scale = 3.0  # m/s (roughly acceptable)
        self.thetadot_scale = 5.0  # rad/s

        self.state: np.ndarray | None = None
        self.reset(seed=seed)

    # ------------------------------
    # Reward shaping (stability oriented)
    # ------------------------------
    def _reward(self, state: np.ndarray, terminated: bool) -> float:
        x, x_dot, theta, theta_dot = state

        # Normalize terms
        x_n = x / self.x_threshold
        th_n = theta / self.theta_threshold_radians
        xd_n = x_dot / self.xdot_scale
        thd_n = theta_dot / self.thetadot_scale

        # Quadratic penalties (LQR-like shaping)
        w_th = 1.0
        w_thd = 0.15
        w_x = 0.1
        w_xd = 0.05

        shaped = 1.0 - (
            w_th * th_n**2 + w_thd * thd_n**2 + w_x * x_n**2 + w_xd * xd_n**2
        )

        # Extra bonus when near the equilibrium (encourage convergence, not only "survive")
        if (
            abs(theta) < 0.02
            and abs(theta_dot) < 0.2
            and abs(x) < 0.05
            and abs(x_dot) < 0.5
        ):
            shaped += 0.5

        # Soft penalty band near limits to discourage drifting
        soft_band = 0.8
        if (
            abs(x) > soft_band * self.x_threshold
            or abs(theta) > soft_band * self.theta_threshold_radians
        ):
            shaped -= 0.2

        # Big penalty if episode terminates due to constraint violation
        if terminated:
            shaped = -10.0

        return float(shaped)

    def step(self, action):
        u = float(
            np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        )
        x, x_dot, theta, theta_dot = self.state

        # Continuous-time dynamics (Euler integration)
        state_dot = self.A @ self.state + self.B * u
        self.state = self.state + self.dt * state_dot

        # Termination by constraints
        terminated = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )

        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps

        reward = self._reward(self.state, terminated)

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomized initial conditions (small angles/velocities)
        # Encourage robustness & avoid "drag-time" strategies
        x0 = self.np_random.uniform(low=-0.05, high=0.05)
        xdot0 = self.np_random.uniform(low=-0.05, high=0.05)
        theta0 = 0.10 + self.np_random.uniform(low=-0.03, high=0.03)  # ~0.1 rad start
        thetadot0 = self.np_random.uniform(low=-0.05, high=0.05)

        self.state = np.array([x0, xdot0, theta0, thetadot0], dtype=np.float64)
        self.step_count = 0
        return np.array(self.state, dtype=np.float32), {}


# ==============================
# 2) Plotting helpers (English labels only)
# ==============================
def plot_training_results(log_folder: str, out_path: str = "rl_training_curve.png"):
    """Plot mean reward during training from SB3 EvalCallback logs."""
    eval_data = np.load(os.path.join(log_folder, "evaluations.npz"))
    timesteps = eval_data["timesteps"]
    mean_rewards = eval_data["results"].mean(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mean_rewards)
    plt.title("Average Reward During Training")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.show()


def plot_simulation_results(
    t: np.ndarray, states: np.ndarray, out_path: str = "rl_simulation_result.png"
):
    """Plot simulated response of the learned policy."""
    x_pos, x_dot, theta, theta_dot = states.T

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(t, np.rad2deg(theta), label=r"$\theta(t)$ (deg)")
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title("RL Controller Response for Inverted Pendulum", fontsize=16)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t, x_pos, label=r"$x(t)$ (m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.show()


# ==============================
# 3) Train & Evaluate
# ==============================
if __name__ == "__main__":
    # Paths
    log_folder = "rl_logs/"
    model_path = os.path.join(log_folder, "best_model")
    os.makedirs(log_folder, exist_ok=True)

    # ----- Training env -----
    train_env = InvertedPendulumEnv()
    train_env = Monitor(train_env)

    # ----- Evaluation env + callback -----
    eval_env = InvertedPendulumEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_folder,
        log_path=log_folder,
        n_eval_episodes=5,
        eval_freq=2000,
        deterministic=True,
        render=False,
    )

    # ----- PPO model -----
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./ppo_pendulum_tensorboard/",
        # Optional but often helpful for continuous control with shaped rewards
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        seed=42,
    )

    # ----- Learn -----
    print("--- Start training ---")
    model.learn(total_timesteps=100_000, callback=eval_callback)
    print("--- Training done ---")

    # ----- Plot training curve -----
    print("\n--- Plot training curve ---")
    plot_training_results(log_folder)

    # ----- Evaluate best model -----
    print("\n--- Evaluate the best model ---")
    best_model = PPO.load(model_path)

    eval_env = InvertedPendulumEnv()  # fresh env for rollout
    obs, info = eval_env.reset()

    states_history = [obs]
    done, truncated = False, False
    while not (done or truncated):
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        states_history.append(obs)

    states_history = np.array(states_history)
    sim_time = np.arange(0, len(states_history) * eval_env.dt, eval_env.dt)

    print(f"Simulation lasted {sim_time[-1]:.2f} seconds.")

    print("\n--- Plot simulated response ---")
    plot_simulation_results(sim_time, states_history)
