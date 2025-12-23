# analyze_pendulum_run.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

# ---------- Minimal copy of the env (must match training) ----------
import gymnasium as gym
from gymnasium import spaces


class InvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dt=0.02, max_episode_steps=2000, reward_cfg=None, seed=None):
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
            low=np.array([-15.0], np.float32),
            high=np.array([15.0], np.float32),
            dtype=np.float32,
        )

        # reward_cfg is unused for analysis rollout (we don't need reward details)
        self.state = None
        self.reset(seed=seed)

    def step(self, action):
        u = float(
            np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        )
        state_dot = self.A @ self.state + self.B * u
        self.state = self.state + self.dt * state_dot

        terminated = bool(
            self.state[0] < -self.x_threshold
            or self.state[0] > self.x_threshold
            or self.state[2] < -self.theta_threshold_radians
            or self.state[2] > self.theta_threshold_radians
        )
        self.step_count += 1
        truncated = self.step_count >= self.max_episode_steps

        # reward value not important for plotting; return 0
        return np.array(self.state, dtype=np.float32), 0.0, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        x0 = self.np_random.uniform(-0.05, 0.05)
        xdot0 = self.np_random.uniform(-0.05, 0.05)
        theta0 = 0.10 + self.np_random.uniform(-0.03, 0.03)
        thetadot0 = self.np_random.uniform(-0.05, 0.05)
        self.state = np.array([x0, xdot0, theta0, thetadot0], dtype=np.float64)
        self.step_count = 0
        return np.array(self.state, dtype=np.float32), {}


# ---------- Plot helpers (English only) ----------
def plot_training_curve(run_dir: str, out_path: str):
    npz_path = os.path.join(run_dir, "evaluations.npz")
    if not os.path.exists(npz_path):
        print(f"[Info] evaluations.npz not found in {run_dir}; skip training plot.")
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
    print(f"[Saved] {out_path}")


def plot_rollout(t, states, out_path: str):
    x, xd, th, thd = states.T
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
    print(f"[Saved] {out_path}")


# ---------- Main (no training, reuse artifacts) ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        required=True,
        help="Directory containing best_model and evaluations.npz",
    )
    ap.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many rollout episodes to average for plotting",
    )
    args = ap.parse_args()

    model_path = os.path.join(args.run_dir, "best_model")
    if not os.path.exists(model_path + ".zip") and not os.path.isdir(model_path):
        # SB3 saves as folder or .zip depending on version/callback
        print(f"[Error] best_model not found under: {args.run_dir}")
        return

    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Plot training curve if available
    plot_training_curve(
        args.run_dir, os.path.join(args.run_dir, "analysis_training_curve.png")
    )

    # Rollout and plot
    env = InvertedPendulumEnv()
    all_states = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = truncated = False
        states = [obs]
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            states.append(obs)
        states = np.array(states)
        all_states.append(states)

    # Use the longest rollout to define time axis by sample count (robust to float)
    max_len = max(s.shape[0] for s in all_states)
    dt = env.dt
    t = np.arange(max_len) * dt  # <-- length-safe time vector

    # If multiple episodes, pad with last value for plotting average (optional)
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

    print(f"Simulation lasted {(states_mean.shape[0] - 1) * dt:.2f} seconds.")
    plot_rollout(t, states_mean, os.path.join(args.run_dir, "analysis_simulation.png"))


if __name__ == "__main__":
    main()
