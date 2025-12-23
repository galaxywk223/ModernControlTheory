# train_pendulum_rl_v3.py
# Goal: reduce steady-state position bias towards 0 without starting from scratch.
# - Loads an existing best_model from --run_dir, then fine-tunes with a bias-fix reward.
# - English-only plots. Handles OMP conflict and time-axis length issues.
# - One script runs end-to-end: load -> finetune -> plots.

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid OpenMP runtime conflict

import argparse
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# ----------------- Minimal env (same 4D observation as before) -----------------
import gymnasium as gym
from gymnasium import spaces


class InvertedPendulumEnv(gym.Env):
    """
    State: [x, x_dot, theta, theta_dot]
    Action: horizontal force (continuous)
    Reward: LQR-like with extra bias-fix terms nudging x -> 0.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt: float = 0.02,
        max_episode_steps: int = 2000,
        reward_cfg: dict | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        self.A = np.array(
            [[0, 1, 0, 0], [0, 0, -0.7171, 0], [0, 0, 0, 1], [0, 0, 31.5512, 0]],
            dtype=np.float64,
        )
        self.B = np.array([0, 0.9756, 0, -2.9268], dtype=np.float64)

        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * np.pi / 180.0

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
        self.action_space = spaces.Box(
            low=np.array([-15.0], dtype=np.float32),
            high=np.array([15.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.xdot_scale = 3.0
        self.thetadot_scale = 5.0

        assert reward_cfg is not None, "reward_cfg required"
        self.cfg = reward_cfg

        self.state = None
        self.reset(seed=seed)

    def _reward(self, state: np.ndarray, u: float, terminated: bool) -> float:
        x, x_dot, theta, theta_dot = state
        # normalized terms
        x_n = x / self.x_threshold
        th_n = theta / self.theta_threshold_radians
        xd_n = x_dot / self.xdot_scale
        thd_n = theta_dot / self.thetadot_scale

        w_th = self.cfg["w_th"]
        w_thd = self.cfg["w_thd"]
        w_x = self.cfg["w_x"]
        w_xd = self.cfg["w_xd"]
        w_u = self.cfg["w_u"]

        shaped = 1.0 - (
            w_th * th_n**2 + w_thd * thd_n**2 + w_x * x_n**2 + w_xd * xd_n**2
        )
        shaped -= w_u * (u**2)

        # persistent |x| penalty to eliminate bias
        shaped -= self.cfg["step_x_penalty"] * abs(x)

        # strong continuous bonus around x≈0 (Gaussian well)
        zb = self.cfg["zero_band"]
        shaped += self.cfg["zero_bonus_gain"] * np.exp(-((x / zb) ** 2))

        # near-equilibrium bonus (tightened)
        nb = self.cfg["near_band"]
        if (
            abs(x) < nb["x"]
            and abs(theta) < nb["th"]
            and abs(x_dot) < nb["xd"]
            and abs(theta_dot) < nb["thd"]
        ):
            shaped += self.cfg["near_bonus"]

        # soft boundary penalty near constraints
        r = self.cfg["soft_band_ratio"]
        if (
            abs(x) > r * self.x_threshold
            or abs(theta) > r * self.theta_threshold_radians
        ):
            shaped -= self.cfg["soft_penalty"]

        if terminated:
            shaped = -10.0
        return float(shaped)

    def step(self, action):
        u = float(
            np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        )
        self.state = self.state + self.dt * (self.A @ self.state + self.B * u)

        terminated = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps
        reward = self._reward(self.state, u, terminated)
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        x0 = self.np_random.uniform(-0.05, 0.05)
        xdot0 = self.np_random.uniform(-0.05, 0.05)
        theta0 = 0.10 + self.np_random.uniform(-0.03, 0.03)
        thetadot0 = self.np_random.uniform(-0.05, 0.05)
        self.state = np.array([x0, xdot0, theta0, thetadot0], dtype=np.float64)
        self.step_count = 0
        return np.array(self.state, dtype=np.float32), {}


# ----------------- Plotting (English only) -----------------
def plot_training_curve(run_dir: str, out_path: str):
    npz_path = os.path.join(run_dir, "evaluations.npz")
    if not os.path.exists(npz_path):
        return
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean_rewards)
    plt.title("Average Reward During Training")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.show()


def plot_rollout(t, states, out_path: str):
    x, xdot, th, thdot = states.T
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax1.plot(t, np.rad2deg(th), label=r"$\theta(t)$ (deg)")
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title("RL Controller Response for Inverted Pendulum", fontsize=18)
    ax1.legend()
    ax1.grid(True)
    ax2.plot(t, x, label=r"$x(t)$ (m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Position (m)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.show()


# ----------------- Bias-fix profile -----------------
BIASFIX = dict(
    w_th=1.0,
    w_thd=0.18,
    w_x=0.75,
    w_xd=0.10,
    w_u=2e-4,
    step_x_penalty=0.05,  # stronger persistent |x|
    zero_bonus_gain=0.25,
    zero_band=0.02,  # reward well centered at x=0 (±2 cm)
    near_bonus=0.9,
    near_band=dict(x=0.02, th=0.015, xd=0.3, thd=0.12),
    soft_band_ratio=0.80,
    soft_penalty=0.25,
)


# ----------------- Utilities -----------------
def detect_model_path(run_dir: str) -> str:
    p_zip = os.path.join(run_dir, "best_model.zip")
    p_dir = os.path.join(run_dir, "best_model")
    if os.path.exists(p_zip):
        return p_zip
    if os.path.isdir(p_dir):
        return p_dir
    raise FileNotFoundError(f"best_model(.zip) not found under: {run_dir}")


# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        required=True,
        help="Directory containing best_model(.zip) and evaluations.npz (previous run).",
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default="runs_v3",
        help="Output root directory for v3 fine-tune.",
    )
    ap.add_argument(
        "--finetune_steps",
        type=int,
        default=30000,
        help="Timesteps for bias-fix fine-tuning.",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Rollout episodes to average for plotting.",
    )
    args = ap.parse_args()

    prev_dir = os.path.abspath(args.run_dir)
    model_path = detect_model_path(prev_dir)

    # output dir
    profile_name = "biasfix"
    out_dir = os.path.join(os.path.abspath(args.out_root), profile_name)
    os.makedirs(out_dir, exist_ok=True)

    # envs
    train_env = Monitor(InvertedPendulumEnv(reward_cfg=BIASFIX))
    eval_env = InvertedPendulumEnv(reward_cfg=BIASFIX)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        n_eval_episodes=5,
        eval_freq=2000,
        deterministic=True,
        render=False,
    )

    # load and fine-tune
    model = PPO.load(model_path, env=train_env, seed=42)
    model.learn(total_timesteps=args.finetune_steps, callback=eval_cb)

    # plots (training curve from this run)
    plot_training_curve(out_dir, os.path.join(out_dir, "training_curve.png"))

    # rollout using the new best model
    best_model = PPO.load(os.path.join(out_dir, "best_model"))
    env = InvertedPendulumEnv(reward_cfg=BIASFIX)
    all_states = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = truncated = False
        states = [obs]
        while not (done or truncated):
            action, _ = best_model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            states.append(obs)
        all_states.append(np.array(states))

    # time axis—safe length
    max_len = max(s.shape[0] for s in all_states)
    t = np.arange(max_len) * env.dt
    if len(all_states) > 1:
        padded = []
        for s in all_states:
            if s.shape[0] < max_len:
                pad = np.repeat(s[-1][None, :], max_len - s.shape[0], axis=0)
                s = np.vstack([s, pad])
            padded.append(s)
        states_mean = np.stack(padded, axis=0).mean(axis=0)
    else:
        states_mean = all_states[0]

    plot_rollout(t, states_mean, os.path.join(out_dir, "simulation_result.png"))

    # quick print of final steady-state x (last second average)
    last_n = int(1.0 / env.dt)
    ss_x = float(states_mean[-last_n:, 0].mean())
    print(f"Estimated steady-state x ≈ {ss_x:.4f} m")


if __name__ == "__main__":
    main()
