import pandas as pd
import numpy as np

# --- 1. 加载数据 ---
try:
    data = pd.read_csv("identification_data.csv")
except FileNotFoundError:
    print("错误: 'identification_data.csv' 未找到。")
    print("请先运行数据采集脚本。")
    exit()

# --- 2. 准备数据 ---
# 计算采样周期
time = data["time"].values
sample_period = time[1] - time[0]

# 提取状态和输入数据
states = data[["x_pos", "x_dot", "theta", "theta_dot"]].values
input_u = data["input_u"].values

# --- 3. 计算状态导数 ---
# 使用np.gradient进行数值微分，axis=0表示对每一列（每个状态）求时间导数
states_dot = np.gradient(states, sample_period, axis=0)

# --- 4. 构建并求解最小二乘问题 (灰盒辨识) ---

# 我们需要辨识的方程是:
# 1. x_dot_dot = A23 * theta + B2 * u
# 2. theta_dot_dot = A43 * theta + B4 * u

# 提取对应的导数作为 y (目标向量)
# y1 对应 x_dot_dot (states_dot的第1列，索引从0开始)
y1 = states_dot[:, 1]
# y2 对应 theta_dot_dot (states_dot的第3列)
y2 = states_dot[:, 3]

# 构建回归矩阵 Phi。对于两个方程，这个矩阵是相同的。
# 矩阵的列是 theta (states的第2列) 和 input_u
Phi = np.vstack([states[:, 2], input_u]).T

# 使用numpy的最小二乘函数求解
# theta1 将会是 [A23_hat, B2_hat]
theta1, _, _, _ = np.linalg.lstsq(Phi, y1, rcond=None)
A23_hat, B2_hat = theta1

# theta2 将会是 [A43_hat, B4_hat]
theta2, _, _, _ = np.linalg.lstsq(Phi, y2, rcond=None)
A43_hat, B4_hat = theta2

# --- 5. 组装辨识出的矩阵 A_hat 和 B_hat ---
A_hat = np.array([[0, 1, 0, 0], [0, 0, A23_hat, 0], [0, 0, 0, 1], [0, 0, A43_hat, 0]])

B_hat = np.array([[0], [B2_hat], [0], [B4_hat]])

# --- 6. 与真实矩阵对比 ---
# 原始的 "真" 矩阵
A_true = np.array([[0, 1, 0, 0], [0, 0, -0.7171, 0], [0, 0, 0, 1], [0, 0, 31.5512, 0]])
B_true = np.array([[0], [0.9756], [0], [-2.9268]])

# 设置打印选项
np.set_printoptions(precision=4, suppress=True)

print("--- 系统辨识结果 ---")
print("\n辨识出的 A 矩阵 (A_hat):")
print(A_hat)
print("\n真实的 A 矩阵 (A_true):")
print(A_true)

print("\n辨识出的 B 矩阵 (B_hat):")
print(B_hat)
print("\n真实的 B 矩阵 (B_true):")
print(B_true)

# --- 7. 计算参数误差 ---
err_A23 = 100 * abs((A23_hat - A_true[1, 2]) / A_true[1, 2])
err_A43 = 100 * abs((A43_hat - A_true[3, 2]) / A_true[3, 2])
err_B2 = 100 * abs((B2_hat - B_true[1, 0]) / B_true[1, 0])
err_B4 = 100 * abs((B4_hat - B_true[3, 0]) / B_true[3, 0])

print("\n--- 参数相对误差 (%) ---")
print(f"A(2,3) Error: {err_A23:.2f}%")
print(f"A(4,3) Error: {err_A43:.2f}%")
print(f"B(2)   Error: {err_B2:.2f}%")
print(f"B(4)   Error: {err_B4:.2f}%")
