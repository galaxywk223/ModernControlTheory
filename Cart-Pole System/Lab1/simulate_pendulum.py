import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- 1. 定义系统矩阵和控制器 (根据您之前的MATLAB输出) ---

# 状态矩阵 A
A = np.array([[0, 1, 0, 0], [0, 0, -0.7171, 0], [0, 0, 0, 1], [0, 0, 31.5512, 0]])

# 输入矩阵 B
B = np.array([[0], [0.9756], [0], [-2.9268]])

# LMI求解器计算出的控制器增益 K
K = np.array([[2.7021, 2.6267, 39.6988, 5.1432]])

# --- 2. 构建闭环系统 ---
# A_cl = A + B*K
# 在numpy中, @ 表示矩阵乘法
A_cl = A + B @ K

print("--- 闭环系统矩阵 A_cl = A + BK ---")
np.set_printoptions(precision=4, suppress=True)
print(A_cl)

# 验证闭环系统的特征值 (应与MATLAB结果一致)
eigenvalues, _ = np.linalg.eig(A_cl)
print("\n--- 闭环系统特征值 ---")
print(eigenvalues)

# --- 3. 设置仿真参数 ---
# 初始条件: [x, x_dot, theta, theta_dot]
# 假设小车和摆杆初始静止，但摆杆有 0.1 弧度 (约 5.7 度) 的初始倾角
x0 = [0, 0, 0.1, 0]

# 仿真时间范围：从 0 到 10 秒
t_span = [0, 20]
# 在指定的时间点上评估结果，以获得平滑的曲线
t_eval = np.linspace(t_span[0], t_span[1], 500)


# --- 4. 定义并求解常微分方程 (ODE) ---
# 定义闭环系统的微分方程: dx/dt = A_cl * x
def closed_loop_system(t, x):
    return A_cl @ x


# 使用 SciPy 的 solve_ivp 求解器
solution = solve_ivp(closed_loop_system, t_span, x0, t_eval=t_eval)

# 提取时间和状态结果
t = solution.t
x_pos, x_dot, theta, theta_dot = solution.y

# --- 5. 绘制仿真结果 ---
plt.style.use("seaborn-v0_8-whitegrid")  # 使用美观的绘图风格
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 子图1: 摆杆角度 (theta) vs. 时间
ax1.plot(t, np.rad2deg(theta), label=r"$\theta(t)$ - 摆杆角度", color="b")
ax1.set_ylabel("角度 (度)")
ax1.set_title("倒立摆系统的闭环响应", fontsize=16)
ax1.legend()
ax1.grid(True)

# 子图2: 小车位置 (x) vs. 时间
ax2.plot(t, x_pos, label=r"$x(t)$ - 小车位置", color="r")
ax2.set_xlabel("时间 (秒)")
ax2.set_ylabel("位置 (米)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()  # 调整布局
# 解决中文显示问题
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.show()
