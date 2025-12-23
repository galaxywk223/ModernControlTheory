# --- SECTION 1: IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- SECTION 2: SYSTEM AND SIMULATION PARAMETERS ---

# System matrices for the DC motor (from paper's Table 1)
A1 = np.array([[-0.479908, 5.1546], [-3.81625, 14.4723]])
B1 = np.array([[5.8705212, 0], [0, 15.50107]])
A2 = np.array([[-1.60261, 9.1632], [-0.5918697, 3.0317]])
B2 = np.array([[10.285129, 0], [0, 2.2282663]])
A3 = np.array([[0.634617, 0.917836], [-0.50569, 2.48116]])
B3 = np.array([[0.7874647, 0], [0, 1.5302844]])
A_list = [A1, A2, A3]
B_list = [B1, B2, B3]

# Disturbance matrices
C1 = np.array([[0.1, 0], [0.1, 0.1]])
C2 = np.array([[0.12, 0.1], [0, 0.1]])
C3 = np.array([[0.1, 0], [0, 0.14]])
C_list = [C1, C2, C3]

# Event-triggering scalar parameters
d = 0.05
sigma = 0.01
delta = 0.1

# Simulation time and initial conditions
t_span = [0, 15]
x0 = np.array([0.14, 0.05])
t_eval = np.linspace(t_span[0], t_span[1], 3000)

# --- SECTION 3: NEWLY CALCULATED CONTROLLER AND TRIGGER MATRICES ---
# These are the values you obtained from the MATLAB LMI solver.

# Newly Calculated Controller Gain Matrices (K)
K1_new = np.array([[-0.4369, 1.2968], [0.2436, -1.3562]])
K2_new = np.array([[-0.0844, -0.0134], [0.5943, -4.7812]])
K3_new = np.array([[-2.6640, 4.5218], [0.9403, -6.5759]])
K_list_new = [K1_new, K2_new, K3_new]

# Newly Calculated Event-Triggering Matrices (Lambda)
Lambda1_new = np.array([[0.7793, -3.6485], [-3.6485, 19.1893]])
Lambda2_new = np.array([[0.2957, -1.3673], [-1.3673, 8.7250]])
Lambda3_new = np.array([[0.3988, -1.5476], [-1.5476, 8.4830]])
Lambda_list_new = [Lambda1_new, Lambda2_new, Lambda3_new]

print(">>> Using newly calculated K and Lambda matrices for simulation. <<<")

# --- SECTION 4: SEMI-MARKOV SWITCHING SIGNAL SIMULATION FUNCTION ---
q_ij = np.array([[0, 0.5, 0.5], [0.2, 0, 0.8], [0.8, 0.2, 0]])
transition_prob = q_ij / q_ij.sum(axis=1, keepdims=True)


def simulate_switching_signal(total_time):
    current_time = 0.0
    current_mode_idx = 2
    time_points = [current_time]
    modes = [current_mode_idx + 1]
    while current_time < total_time:
        if current_mode_idx == 0:
            sojourn_time = np.random.weibull(a=2)
        elif current_mode_idx == 1:
            sojourn_time = np.random.weibull(a=3)
        else:  # mode_idx == 2
            sojourn_time = np.random.exponential(scale=1.0 / 0.5)
        current_time += sojourn_time
        possible_next_modes = [0, 1, 2]
        probabilities = transition_prob[current_mode_idx]
        next_mode_idx = np.random.choice(possible_next_modes, p=probabilities)
        time_points.append(current_time)
        modes.append(current_mode_idx + 1)
        current_mode_idx = next_mode_idx
    return np.array(time_points), np.array(modes)


# --- SECTION 5: THE SYSTEM SIMULATOR CLASS ---
class SystemSimulator:
    def __init__(self, A, B, C, K, Lambda, switch_t, switch_m, d, sigma, delta):
        self.A_list, self.B_list, self.C_list = A, B, C
        self.K_list, self.Lambda_list = K, Lambda
        self.switch_times, self.switch_modes = switch_t, switch_m
        self.d, self.sigma, self.delta = d, sigma, delta
        self.last_trigger_time = -1
        self.x_triggered = None
        self.trigger_times_history = []

    def get_mode_idx(self, t):
        idx = np.searchsorted(self.switch_times, t, side="right")
        return self.switch_modes[idx - 1] - 1

    def disturbance(self, t):
        return np.array([np.exp(-t), np.exp(-t)])

    def dynamics(self, t, x):
        current_mode_idx = self.get_mode_idx(t)
        trigger_event = False
        if self.last_trigger_time == -1:
            trigger_event = True
        elif t >= self.last_trigger_time + self.d:
            error = x - self.x_triggered
            Lambda_i = self.Lambda_list[current_mode_idx]
            lhs = error.T @ Lambda_i @ error
            w_t = self.disturbance(t)
            rhs = self.sigma * (x.T @ Lambda_i @ x) + self.delta * (w_t.T @ w_t)
            if lhs >= rhs:
                trigger_event = True
        if trigger_event:
            self.last_trigger_time = t
            self.x_triggered = x
            self.trigger_times_history.append(t)
        K_i = self.K_list[current_mode_idx]
        u = K_i @ self.x_triggered
        A_i, B_i, C_i = (
            self.A_list[current_mode_idx],
            self.B_list[current_mode_idx],
            self.C_list[current_mode_idx],
        )
        w_t = self.disturbance(t)
        dxdt = A_i @ x + B_i @ u + C_i @ w_t
        return dxdt


# --- SECTION 6: MAIN SIMULATION EXECUTION ---
np.random.seed(42)
simulation_time_points, simulation_modes = simulate_switching_signal(t_span[1])

simulator = SystemSimulator(
    A=A_list,
    B=B_list,
    C=C_list,
    K=K_list_new,
    Lambda=Lambda_list_new,  # <-- Using the new matrices here!
    switch_t=simulation_time_points,
    switch_m=simulation_modes,
    d=d,
    sigma=sigma,
    delta=delta,
)

print("Running simulation with the new controller...")
sol = solve_ivp(
    fun=simulator.dynamics,
    t_span=t_span,
    y0=x0,
    t_eval=t_eval,
    method="RK45",
    max_step=0.01,
)
print("Simulation complete!")
print(f"Total event triggers: {len(simulator.trigger_times_history)}")

# --- SECTION 7: VISUALIZATION ---
state_trajectory = sol.y.T
time_vector = sol.t

# Plot 1: State Trajectories and Control Inputs
trigger_times = simulator.trigger_times_history
u_values_at_triggers = []
for t_trig in trigger_times:
    idx = np.searchsorted(time_vector, t_trig)
    x_s = state_trajectory[idx]
    mode_idx_s = simulator.get_mode_idx(t_trig)
    u_s = K_list_new[mode_idx_s] @ x_s
    u_values_at_triggers.append(u_s)
u_values_at_triggers = np.array(u_values_at_triggers)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_vector, state_trajectory[:, 0], label="$x_1(t)$ (angular velocity)")
ax.plot(time_vector, state_trajectory[:, 1], label="$x_2(t)$ (current)")
ax.set_title("State Trajectories using OUR Calculated Controller")
ax.set_xlabel("Time (s)")
ax.set_ylabel("State values")
ax.legend()
ax.grid(True)
ax.axhline(0, color="black", lw=0.5)
ax_inset = fig.add_axes([0.4, 0.5, 0.3, 0.3])
ax_inset.step(trigger_times, u_values_at_triggers[:, 0], where="post", label="$u_1(t)$")
ax_inset.step(trigger_times, u_values_at_triggers[:, 1], where="post", label="$u_2(t)$")
ax_inset.set_title("Control Inputs u(t)")
ax_inset.grid(True)
plt.show()

# Plot 2: Release Intervals
release_intervals = np.diff(trigger_times)
release_instants = trigger_times[1:]
plt.figure(figsize=(10, 5))
plt.stem(release_instants, release_intervals)
plt.title("Release Intervals using OUR Calculated Triggering Matrix")
plt.xlabel("Time (s)")
plt.ylabel("Interval length (s)")
plt.axhline(d, color="red", linestyle=":", lw=1.5, label=f"Min interval d={d}")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
