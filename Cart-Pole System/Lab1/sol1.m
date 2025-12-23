% =========================================================================
% 使用 LMI 判断倒立摆系统状态反馈控制器的可行性
% -------------------------------------------------------------------------
% 需要: YALMIP 工具箱 (https://yalmip.github.io/) 和一个 SDP 求解器 (如 SeDuMi)
% =========================================================================

% --- 1. 清理工作区 ---
clear;
clc;
close all;

% --- 2. 定义系统矩阵 (根据您提供的数值) ---
A = [0, 1, 0, 0;
     0, 0, -0.7171, 0;
     0, 0, 0, 1;
     0, 0, 31.5512, 0];

B = [0;
     0.9756;
     0;
     -2.9268];

% 获取系统维度
% n = 状态数量, m = 输入数量
[n, m] = size(B);

% --- 3. 定义 LMI 决策变量 ---
% 定义一个 n x n 的对称正定矩阵 Q
Q = sdpvar(n, n, 'symmetric');
% 定义一个 m x n 的矩阵 Y
Y = sdpvar(m, n);

% --- 4. 定义 LMI 约束条件 ---
% 创建约束列表
Constraints = [];

% 约束1: Q 必须是正定的 (Q > 0)
Constraints = [Constraints, Q >= 1e-6 * eye(n)]; % 使用一个小的正数确保严格正定

% 约束2: 李雅普诺夫不等式 A*Q + B*Y + (A*Q + B*Y)' < 0
LMI_stability = A*Q + B*Y + (A*Q + B*Y)';
Constraints = [Constraints, LMI_stability <= -1e-6 * eye(n)]; % 使用一个小的负数确保严格负定

% --- 5. 设置求解器选项 ---
% 'verbose', 1 会显示求解过程
options = sdpsettings('verbose', 1, 'solver', 'sedumi'); % 您也可以尝试 'sdpt3'

% --- 6. 求解 LMI ---
solution = solvesdp(Constraints, [], options);

% --- 7. 判断解的可行性并输出结果 ---
if solution.problem == 0
    fprintf('\n==================================\n');
    fprintf('  LMI 问题是可行的 (Feasible)！\n');
    fprintf('==================================\n');
    fprintf('这意味着存在一个稳定的状态反馈控制器 K。\n\n');
    
    % 从 LMI 的解中恢复出控制器增益 K
    % 因为 Y = K*Q, 所以 K = Y * inv(Q)
    Q_sol = value(Q);
    Y_sol = value(Y);
    K = Y_sol / Q_sol; % 在MATLAB中, Y/Q 比 Y*inv(Q) 数值上更稳定
    
    fprintf('计算得到的控制器增益 K 为:\n');
    disp(K);
    
    % --- 验证控制器 ---
    A_cl = A + B*K;
    eigenvalues_cl = eig(A_cl);
    
    fprintf('闭环系统 A_cl = (A + BK) 的特征值为:\n');
    disp(eigenvalues_cl);
    
    if all(real(eigenvalues_cl) < 0)
        fprintf('\n验证成功：所有特征值实部均为负，闭环系统稳定。\n');
    else
        fprintf('\n验证失败：存在非负实部的特征值，检查计算过程。\n');
    end
    
else
    fprintf('\n=====================================\n');
    fprintf('  LMI 问题是不可行的 (Infeasible)。\n');
    fprintf('=====================================\n');
    fprintf('求解器报告的问题代码: %s\n', solution.info);
    fprintf('这意味着无法找到满足条件的控制器，请检查模型或约束。\n');
end