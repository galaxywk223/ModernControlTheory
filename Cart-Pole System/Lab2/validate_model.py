import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# --- 1. 加载辨识出的模型、控制器和验证数据 ---

# 这是您在第二步中辨识出的矩阵
A_hat = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -0.1143, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 29.8034, 0.0],
    ]
)

B_hat = np.array([[0.0], [0.9392], [0.0], [-2.8232]])

# 原始控制器增益K
K = np.array([[2.7021, 2.6267, 39.6988, 5.1432]])

# 加载用于验证的原始数据
try:
    data = pd.read_csv("identification_data.csv")
except FileNotFoundError:
    print("错误: 'identification_data.csv' 未找到。")
    exit()

time_vec = data["time"].values
states_measured = data[["x_pos", "x_dot", "theta", "theta_dot"]].values


# --- 2. 重新生成与实验中完全相同的激励信号 r(t) ---
# 需要从数据采集脚本中复制PRBS生成函数和参数
def generate_prbs(amplitude, clock_period, num_clocks):
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


T_END = 20
PRBS_AMPLITUDE = 0.5
PRBS_CLOCK = 0.2
prbs_t, prbs_r = generate_prbs(PRBS_AMPLITUDE, PRBS_CLOCK, T_END / PRBS_CLOCK)
r_func = interp1d(prbs_t, prbs_r, kind="previous", fill_value="extrapolate")


# --- 3. 设置并运行闭环仿真 ---
# 构建基于辨识模型的闭环矩阵 A_cl_hat = A_hat + B_hat * K
A_cl_hat = A_hat + B_hat @ K


# 定义辨识出的闭环系统的动力学
def identified_closed_loop_dynamics(t, x):
    r_t = r_func(t)
    # 动态 = A_cl_hat * x + B_hat * r(t)
    return A_cl_hat @ x + B_hat.flatten() * r_t


# 仿真的初始条件与真实数据一致
x0 = states_measured[0, :]

# 运行仿真
solution = solve_ivp(
    identified_closed_loop_dynamics, [time_vec[0], time_vec[-1]], x0, t_eval=time_vec
)
states_predicted = solution.y.T


# --- 4. 结果可视化与量化评估 ---
def calculate_fit_percentage(y_true, y_pred):
    nrmse = np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true - np.mean(y_true))
    return (1 - nrmse) * 100


fit_theta = calculate_fit_percentage(states_measured[:, 2], states_predicted[:, 2])
fit_x_pos = calculate_fit_percentage(states_measured[:, 0], states_predicted[:, 0])

print("--- 模型验证结果 (闭环修正) ---")
print(f"摆杆角度 (theta) 的拟合优度: {fit_theta:.2f}%")
print(f"小车位置 (x_pos) 的拟合优度: {fit_x_pos:.2f}%")

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Model Validation (Closed-Loop): Measured vs. Predicted", fontsize=16)

axes[0].plot(
    time_vec, np.rad2deg(states_measured[:, 2]), "b-", label="Measured (Real Data)"
)
axes[0].plot(
    time_vec,
    np.rad2deg(states_predicted[:, 2]),
    "r--",
    label=f"Predicted (Identified Model) - Fit: {fit_theta:.2f}%",
)
axes[0].set_ylabel("Pendulum Angle (deg)")
axes[0].legend()

axes[1].plot(time_vec, states_measured[:, 0], "b-", label="Measured (Real Data)")
axes[1].plot(
    time_vec,
    states_measured[:, 0],
    "r--",
    label=f"Predicted (Identified Model) - Fit: {fit_x_pos:.2f}%",
)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Cart Position (m)")
axes[1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("model_validation_plot_corrected.png", dpi=300)
print("\n--- 修正后的验证图表已保存到 model_validation_plot_corrected.png ---")
plt.show()
