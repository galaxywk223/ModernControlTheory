%% zIE_06_sim_ET_hatw.m —— 事件触发仿真（含 hat{w} 项）
clear; clc;
load zIE_sys.mat
load zIE_stage0.mat
load zIE_stageB.mat

rng(2);
T = 5; dt = 1e-3; Nstep = round(T / dt);
sigma_sim = sigma; % 与设计一致
delta_sim = delta; % 启用 hat{w} 权重
hmin_steps = 3;

% 马尔可夫跳转
Pi = [-1.0 0.6 0.4; 0.5 -1.2 0.7; 0.4 0.5 -0.9];
Pstep = eye(N) + Pi * dt; Pstep = max(Pstep, 0);
for r = 1:N, Pstep(r, :) = Pstep(r, :) / sum(Pstep(r, :)); end

% —— 简单的 \hat w 估计器（残差驱动的一阶滤波器）
%   \dot{\hat w} = -lambda \hat w + Lw * (y - C x_hold - D u)
%   注意：这是一个“实用近似器”，不改变你前面求解的 LMI；Lw 可调小一些以免过激。
Lw = 1e-2 * eye(q);

x = [0.2; -0.1]; rk = 1; x_hold = x; u = zeros(m, 1);
w_hat = zeros(q, 1);

events = 1; ev_k = zeros(Nstep, 1); ev_k(1) = 1;
    Xlog = zeros(n, Nstep); Ulog = zeros(m, Nstep); Rlog = zeros(1, Nstep);
    Whatlog = zeros(q, Nstep); Tlog = (0:Nstep - 1) * dt;

    for k = 1:Nstep
        t = Tlog(k);

        % Markov 切换
        rk = find(rand <= cumsum(Pstep(rk, :)), 1);

        Ki = Kv{rk}; Lam = Lambda_eff{rk}; Ci = C{rk}; Di = D{rk};

        % 控制（ZOH）
        u = Ki * x_hold;

        % 输出与残差（用保持的 x_hold 做观测预测）
        y = Ci * x + Di * u;
        y_pred = Ci * x_hold + Di * u;
        res = y - y_pred;

        % \hat w 动态
        w_hat = w_hat + dt * (-lambda * w_hat + Lw * res);

        % 状态推进
        xdot = A{rk} * x + B{rk} * u;
        x = x + dt * xdot;

        % 事件判据
        e = x_hold - x;
        lhs = e.' * Lam * e;
        rhs = sigma_sim * (x.' * Lam * x) + delta_sim * (w_hat.' * Ghv{rk} * w_hat);

        if (k - ev_k(events)) >= hmin_steps && lhs >= rhs
            x_hold = x; % 采样

            events = events + 1; ev_k(events) = k;
            end

            % 记录
            Xlog(:, k) = x; Ulog(:, k) = u; Rlog(k) = rk; Whatlog(:, k) = w_hat;
        end

        ev_k = ev_k(1:events);

        % 统计与图
        fprintf('[SIM+hW] events = %d over %.2fs (avg %.1f Hz)\n', events, T, events / T);

        if numel(ev_k) >= 2
            dti = diff(Tlog(ev_k));
            fprintf('         median dT = %.3f s | min/mean/max = %.3f / %.3f / %.3f s\n', ...
                median(dti), min(dti), mean(dti), max(dti));
        else
            dti = [];
        end

        figure; histogram(dti, 30); grid on; xlabel('\Delta t [s]'); ylabel('count');
        title('Inter-event interval (with \hat{w})');

        figure; plot(Tlog, Xlog); grid on; xlabel('t [s]'); ylabel('x'); title('States');
        figure; plot(Tlog, Ulog); grid on; xlabel('t [s]'); ylabel('u'); title('Control');
        figure; stairs(Tlog, Rlog); grid on; xlabel('t [s]'); ylabel('mode'); title('Mode r(t)');
        figure; plot(Tlog, Whatlog); grid on; xlabel('t [s]'); ylabel('\hat{w}'); title('w-hat');

        % 只存数值
        save zIE_sim_hatw.mat T dt sigma_sim delta_sim Xlog Ulog Rlog ev_k dti Whatlog -v7
        disp('zIE_sim_hatw.mat 已生成');
