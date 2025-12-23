import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. System Matrices and Controller ---
A = np.array([[0, 1, 0, 0], [0, 0, -0.7171, 0], [0, 0, 0, 1], [0, 0, 31.5512, 0]])
B = np.array([[0], [0.9756], [0], [-2.9268]])
K = np.array([[2.7021, 2.6267, 39.6988, 5.1432]])


# --- 2. PRBS Signal Generation Function ---
def generate_prbs(amplitude, clock_period, num_clocks):
    """
    Generates a Pseudo-Random Binary Sequence (PRBS).
    Uses a 10-bit Linear Feedback Shift Register (LFSR) with taps at 10 and 3.
    """
    num_steps = int(num_clocks)
    register = np.ones(10, dtype=int)
    prbs_signal = []

    for _ in range(num_steps):
        prbs_signal.append(register[-1])
        new_bit = register[-1] ^ register[-3]
        register[1:] = register[:-1]
        register[0] = new_bit

    prbs_signal = np.array(prbs_signal) * 2 * amplitude - amplitude
    t = np.arange(0, num_clocks * clock_period, clock_period)

    return t, prbs_signal


# --- 3. Simulation and Identification Parameters ---
T_START = 0
T_END = 20
SAMPLE_RATE = 50
SAMPLE_PERIOD = 1.0 / SAMPLE_RATE

PRBS_AMPLITUDE = 0.5
PRBS_CLOCK = 0.2

prbs_t, prbs_r = generate_prbs(PRBS_AMPLITUDE, PRBS_CLOCK, T_END / PRBS_CLOCK)
r_func = interp1d(prbs_t, prbs_r, kind="previous", fill_value="extrapolate")

x0 = [0, 0, 0, 0]
t_eval = np.arange(T_START, T_END, SAMPLE_PERIOD)

# --- 4. System Dynamics for Identification (Corrected) ---
input_log = []
time_log = []


def system_for_identification(t, x):
    """
    System dynamics including closed-loop control and external excitation r(t).
    """
    r_t = r_func(t)
    u_control = (K @ x).item()  # Corrected sign to match stable model
    u_t = u_control + r_t

    time_log.append(t)
    input_log.append(u_t)

    x_dot = A @ x + B.flatten() * u_t

    return x_dot


# --- 5. Run Simulation and Process Data ---
solution = solve_ivp(
    system_for_identification,
    [T_START, T_END],
    x0,
    t_eval=t_eval,
    max_step=SAMPLE_PERIOD,
)

states = solution.y.T
times = solution.t

u_interp_func = interp1d(time_log, input_log, kind="linear", fill_value="extrapolate")
final_u = u_interp_func(times)

data = pd.DataFrame(
    {
        "time": times,
        "x_pos": states[:, 0],
        "x_dot": states[:, 1],
        "theta": states[:, 2],
        "theta_dot": states[:, 3],
        "input_u": final_u,
    }
)

# Save data to CSV
data.to_csv("identification_data.csv", index=False)
print("--- Data collected and saved to identification_data.csv ---")
print(data.head())

# --- 6. Plot and Save the Results ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Subplot 1: Excitation Signal r(t) and Total Input u(t)
axes[0].plot(
    times,
    r_func(times),
    "g--",
    label=r"Excitation Signal $r(t)$ (PRBS)",
    drawstyle="steps-post",
)
axes[0].plot(times, data["input_u"], "r-", label=r"Total Input $u(t) = Kx(t) + r(t)$")
axes[0].set_ylabel("Force (N)")
axes[0].set_title("Collected System Identification Data (Corrected)", fontsize=16)
axes[0].legend()
axes[0].grid(True)

# Subplot 2: Pendulum Angle
axes[1].plot(
    times, np.rad2deg(data["theta"]), "b-", label=r"$\theta(t)$ - Pendulum Angle"
)
axes[1].set_ylabel("Angle (deg)")
axes[1].legend()
axes[1].grid(True)

# Subplot 3: Cart Position
axes[2].plot(times, data["x_pos"], "m-", label=r"$x(t)$ - Cart Position")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Position (m)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()

# Save the figure to a file
plt.savefig("data_collection_plot.png", dpi=300)
print("\n--- Plot saved to data_collection_plot.png ---")

# Show the plot
plt.show()
