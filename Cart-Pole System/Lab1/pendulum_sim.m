%% pendulum_sim.m
% 线性化倒立摆（小角度、直立平衡点）闭环仿真 + 动画
clear; clc; close all;

%% 参数
M = 1.0;      % 小车质量 [kg]
m = 0.1;      % 摆杆质量 [kg]
L = 0.5;      % 摆杆长度 [m]
g = 9.8;      % 重力 [m/s^2]
I = 0;        % 若未给转动惯量，常见线化模型取 I≈0（质点杆）
l = L;        % 与题面符号一致（用 L 作为 l）

Delta = I*(M+m) + M*m*l^2;   % 题面给定 Δ

%% 状态空间矩阵（题面给定的线化模型）
A = [ 0      1      0           0;
      0      0   -(m^2*g*l^2)/Delta   0;
      0      0      0           1;
      0      0  (m*g*l*(M+m))/Delta   0 ];

B = [ 0;
     (I + m*l^2)/Delta;
      0;
     -m*l/Delta ];

% 题面给定数值代入后 A、B（留作核对）
A_num = [0 1 0 0; 0 0 -0.7171 0; 0 0 0 1; 0 0 31.5512 0];
B_num = [0; 0.9756; 0; -2.9268];

% 可切换使用题面直接给出的数值矩阵，避免数值微小差异
use_given_numeric_AB = true;
if use_given_numeric_AB
    A = A_num;
    B = B_num;
end

%% 控制增益（题面通过 YALMIP 得到）
K = -[2.7021  2.6267  39.6988  5.1432];

%% 初始状态与仿真设置
x0 = [0.5; 0.4; 0; 0];        % [位置 x, 速度 xdot, 角度 theta(弧度), 角速度 thetadot]
tspan = [0 100];             % 仿真 10 s

% 将参数打包，传给 ODE
p.A = A; p.B = B; p.K = K;

%% 数值积分（闭环：u = -Kx）
% 状态方程： xdot = (A - B*K) x
odefun = @(t,x) cartpole_lin_closedloop(t, x, p);
opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[t, X] = ode45(odefun, tspan, x0, opts);

% 计算控制输入
U = -(K*X.').';   % 每行对应时刻的 u(t)

%% 画图：状态与控制输入
figure('Color','w','Name','States & Control');
subplot(5,1,1); plot(t, X(:,1),'LineWidth',1.5); grid on; ylabel('x [m]');
subplot(5,1,2); plot(t, X(:,2),'LineWidth',1.5); grid on; ylabel('ẋ [m/s]');
subplot(5,1,3); plot(t, X(:,3),'LineWidth',1.5); grid on; ylabel('\theta [rad]');
subplot(5,1,4); plot(t, X(:,4),'LineWidth',1.5); grid on; ylabel('\thetȧ [rad/s]');
subplot(5,1,5); plot(t, U,'LineWidth',1.5); grid on; ylabel('u [N]'); xlabel('t [s]');
sgtitle('Closed-loop response (u = -Kx)');

%% 动画
anim.opt.dt_visual = 0.01;     % 可视化步长（插值）
anim.track_width = 0.3;        % 小车宽
anim.track_height = 0.15;      % 小车高
anim.rod_len = L;              % 动画用杆长（与模型一致）
anim.x_lim = 1.5;              % 画面世界范围（左右）
anim.y_base = -0.2;            % 轨道高度
animate_cartpole(t, X, anim);
