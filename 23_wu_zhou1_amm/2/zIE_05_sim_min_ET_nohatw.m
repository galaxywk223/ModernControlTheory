%% zIE_05_sim_min_ET_nohatw.m —— 最小仿真（不含 hat{w} 触发项）
clear; clc;
load zIE_sys.mat
load zIE_stage0.mat
load zIE_stageB.mat

rng(1);
T = 5; dt = 1e-3; Nstep = round(T / dt);
sigma_sim = 0.01; delta_sim = 0.0; % 这里先不启用 hat{w}，只留 x-触发
hmin_steps = 3; % 最小事件间隔步数（反 Zeno）

% 马尔可夫跳转（常值生成元 → 离散步近似）
Pi = [-1.0 0.6 0.4; 0.5 -1.2 0.7; 0.4 0.5 -0.9];
Pstep = eye(N) + Pi * dt; Pstep = max(Pstep, 0);
for r = 1:N, Pstep(r, :) = Pstep(r, :) / sum(Pstep(r, :)); end

x0 = [0.2; -0.1];
x = x0; rk = 1; x_hold = x; u = zeros(m, 1);

events = 1; ev_k = zeros(Nstep, 1); ev_k(1) = 1;
    Xlog = zeros(n, Nstep); Ulog = zeros(m, Nstep);
    Rlog = zeros(1, Nstep); Tlog = (0:Nstep - 1) * dt;

    for k = 1:Nstep
        t = Tlog(k);
        rk = find(rand <= cumsum(Pstep(rk, :)), 1);

        Ki = Kv{rk}; Lam = Lambda_eff{rk};
        u = Ki * x_hold;

        xdot = A{rk} * x + B{rk} * u; % 无显式扰动
        x = x + dt * xdot;

        e = x_hold - x;
        lhs = e.' * Lam * e;
        rhs = sigma_sim * (x.' * Lam * x) + delta_sim * 0;

        if (k - ev_k(events)) >= hmin_steps && lhs >= rhs
            x_hold = x;

            events = events + 1; ev_k(events) = k;
            end

            Xlog(:, k) = x; Ulog(:, k) = u; Rlog(k) = rk;
        end

        ev_k = ev_k(1:events);
        fprintf('[SIM] events = %d over %.2fs (avg %.1f Hz)\n', events, T, events / T);

        % 事件间隔统计
        if numel(ev_k) >= 2
            dti = diff(Tlog(ev_k));
            fprintf('      median dT = %.3f s | min/mean/max = %.3f / %.3f / %.3f s\n', ...
                median(dti), min(dti), mean(dti), max(dti));
        else
            dti = [];
        end

        % 画图
        figure; histogram(dti, 30); grid on; xlabel('\Delta t between events [s]'); ylabel('count');
        title('Inter-event interval histogram');

        figure; plot(Tlog, Xlog); grid on; xlabel('t [s]'); ylabel('x'); title('States');
        figure; plot(Tlog, Ulog); grid on; xlabel('t [s]'); ylabel('u'); title('Control');
        figure; stairs(Tlog, Rlog); grid on; xlabel('t [s]'); ylabel('mode'); title('Mode r(t)');
        figure; stem(Tlog(ev_k), ones(numel(ev_k), 1)); grid on; xlabel('t [s]'); title('Event times');

        % 只存数值
        save zIE_sim.mat T dt Xlog Ulog Rlog ev_k dti -v7
        disp('zIE_sim.mat 已生成');
