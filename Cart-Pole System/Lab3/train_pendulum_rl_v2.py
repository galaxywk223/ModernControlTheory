import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


# ==============================
# Config profiles
# ==============================
PROFILES = {
    # More cautious weighting; focuses on stabilizing angle with meaningful
    # position drive, small action penalty. Tends to small residual x.
    "conservative": dict(
        w_th=1.0,
        w_thd=0.15,
        w_x=0.35,
        w_xd=0.07,
        w_u=1e-4,
        near_bonus=0.6,  # extra reward when close to equilibrium
        near_band=dict(x=0.05, th=0.02, xd=0.5, thd=0.2),
        soft_band_ratio=0.85,  # start soft penalty when close to limits
        soft_penalty=0.2,
        step_x_penalty=0.01,  # per-step |x| penalty to remove steady-state bias
        total_timesteps=120_000,
    ),
    # Stronger position weight and bonus, slightly larger per-step |x| penalty.
    # Returns to x=0 faster, at the cost of more control effort.
    "aggressive": dict(
        w_th=1.0,
        w_thd=0.18,
        w_x=0.55,
        w_xd=0.10,
        w_u=2e-4,
        near_bonus=0.8,
        near_band=dict(x=0.03, th=0.015, xd=0.4, thd=0.15),
        soft_band_ratio=0.80,
        soft_penalty=0.25,
        step_x_penalty=0.02,
        total_timesteps=150_000,
    ),
}


# ==============================
# Environment
# ==============================
class InvertedPendulumEnv(gym.Env):
    """
    Inverted pendulum (cart + pole) linearized model.
    State  : [x, x_dot, theta, theta_dot]
    Action : horizontal force (continuous)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.02,
        max_episode_steps: int = 2000,  # 40 s
        reward_cfg: dict | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # Linearized continuous-time dynamics: x_dot = A x + B u
        self.A = np.array(
            [[0, 1, 0, 0], [0, 0, -0.7171, 0], [0, 0, 0, 1], [0, 0, 31.5512, 0]],
            dtype=np.float64,
        )
        self.B = np.array([0, 0.9756, 0, -2.9268], dtype=np.float64)

        # Limits
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

        # Action space
        self.action_space = spaces.Box(
            low=np.array([-15.0], dtype=np.float32),
            high=np.array([15.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Scales for normalization
        self.xdot_scale = 3.0  # m/s
        self.thetadot_scale = 5.0  # rad/s

        # Reward config
        assert reward_cfg is not None, "reward_cfg must be provided"
        self.cfg = reward_cfg

        self.state = None
        self.reset(seed=seed)

    # Reward shaping focusing on real stabilization and x->0
    def _reward(
        self, prev_state: np.ndarray, state: np.ndarray, u: float, terminated: bool
    ) -> float:
        x, x_dot, theta, theta_dot = state

        # Normalize
        x_n = x / self.x_threshold
        th_n = theta / self.theta_threshold_radians
        xd_n = x_dot / self.xdot_scale
        thd_n = theta_dot / self.thetadot_scale

        w_th = self.cfg["w_th"]
        w_thd = self.cfg["w_thd"]
        w_x = self.cfg["w_x"]
        w_xd = self.cfg["w_xd"]
        w_u = self.cfg["w_u"]
        near_b = self.cfg["near_bonus"]
        near = self.cfg["near_band"]
        soft_r = self.cfg["soft_band_ratio"]
        soft_p = self.cfg["soft_penalty"]
        step_x = self.cfg["step_x_penalty"]

        # Quadratic penalties (LQR-like)
        shaped = 1.0 - (
            w_th * th_n**2 + w_thd * thd_n**2 + w_x * x_n**2 + w_xd * xd_n**2
        )

        # Small action (energy) penalty to avoid large kicks that push x away
        shaped -= w_u * (u**2)

        # Persistent |x| penalty to remove steady-state bias
        shaped -= step_x * abs(x)

        # Extra bonus when near the equilibrium (encourage convergence to (0,0,0,0))
        if (
            abs(x) < near["x"]
            and abs(theta) < near["th"]
            and abs(x_dot) < near["xd"]
            and abs(theta_dot) < near["thd"]
        ):
            shaped += near_b

        # Soft penalty band near limits
        if (
            abs(x) > soft_r * self.x_threshold
            or abs(theta) > soft_r * self.theta_threshold_radians
        ):
            shaped -= soft_p

        # Big penalty if terminated by constraints
        if terminated:
            shaped = -10.0

        return float(shaped)

    def step(self, action):
        u = float(
            np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        )

        prev_state = self.state.copy()
        state_dot = self.A @ self.state + self.B * u
        self.state = self.state + self.dt * state_dot

        # Termination
        terminated = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )

        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps

        reward = self._reward(prev_state, self.state, u, terminated)
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomized small initial conditions
        x0 = self.np_random.uniform(-0.05, 0.05)
        xdot0 = self.np_random.uniform(-0.05, 0.05)
        theta0 = 0.10 + self.np_random.uniform(-0.03, 0.03)
        thetadot0 = self.np_random.uniform(-0.05, 0.05)

        self.state = np.array([x0, xdot0, theta0, thetadot0], dtype=np.float64)
        self.step_count = 0
        return np.array(self.state, dtype=np.float32), {}


# ==============================
# Plotting (English labels only)
# ==============================
def plot_training_results(log_folder: str, out_path: str):
    data_path = os.path.join(log_folder, "evaluations.npz")
    eval_data = np.load(data_path)
    timesteps = eval_data["timesteps"]
    mean_rewards = eval_data["results"].mean(axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean_rewards)
    plt.title("Average Reward During Training")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.show()


def plot_simulation_results(t, states, out_path: str):
    x_pos, x_dot, theta, theta_dot = states.T

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax1.plot(t, np.rad2deg(theta), label=r"$\theta(t)$ (deg)")
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title("RL Controller Response for Inverted Pendulum", fontsize=18)
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
# Train & Evaluate
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="conservative",
        help="Reward profile to use.",
    )
    parser.add_argument(
        "--out_root", type=str, default="runs_v2", help="Root output directory."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PROFILES[args.profile]

    # Output directories per profile
    out_dir = os.path.join(args.out_root, args.profile)
    os.makedirs(out_dir, exist_ok=True)

    log_folder = out_dir
    model_path = os.path.join(out_dir, "best_model")
    train_plot_path = os.path.join(out_dir, "training_curve.png")
    sim_plot_path = os.path.join(out_dir, "simulation_result.png")

    # ----- Envs -----
    train_env = InvertedPendulumEnv(reward_cfg=cfg)
    train_env = Monitor(train_env)
    eval_env = InvertedPendulumEnv(reward_cfg=cfg)

    # ----- Callback -----
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        n_eval_episodes=5,
        eval_freq=2000,
        deterministic=True,
        render=False,
    )

    # ----- PPO -----
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(out_dir, "tb"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        seed=args.seed,
    )

    # ----- Train -----
    print(f"--- Start training [{args.profile}] ---")
    model.learn(total_timesteps=cfg["total_timesteps"], callback=eval_callback)
    print(f"--- Training done. Artifacts in: {out_dir} ---")

    # ----- Plot training curve -----
    print("\n--- Plot training curve ---")
    plot_training_results(log_folder, train_plot_path)

    # ----- Evaluate best model -----
    print("\n--- Evaluate the best model ---")
    best_model = PPO.load(model_path)

    rollout_env = InvertedPendulumEnv(reward_cfg=cfg)  # fresh env
    obs, info = rollout_env.reset()

    states = [obs]
    done = truncated = False
    while not (done or truncated):
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = rollout_env.step(action)
        states.append(obs)

    states = np.array(states)
    sim_time = np.arange(0, len(states) * rollout_env.dt, rollout_env.dt)

    print(f"Simulation lasted {sim_time[-1]:.2f} seconds.")
    print("--- Plot simulation response ---")
    plot_simulation_results(sim_time, states, sim_plot_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved training plot to: {train_plot_path}")
    print(f"Saved simulation plot to: {sim_plot_path}")


if __name__ == "__main__":
    main()
