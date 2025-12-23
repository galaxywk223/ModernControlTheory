%% zIE_07_sweep_sigma.m —— 扫描 sigma & hmin，画折衷曲线
clear; clc;
load zIE_sys.mat
load zIE_stage0.mat
load zIE_stageB.mat

rng(3);
T = 5; dt = 1e-3; Nstep = round(T / dt);

sigma_list = [0.005 0.01 0.02 0.05 0.1];
hmin_list = [1 3 5];

Pi = [-1.0 0.6 0.4; 0.5 -1.2 0.7; 0.4 0.5 -0.9];
Pstep = eye(N) + Pi * dt; Pstep = max(Pstep, 0);
for r = 1:N, Pstep(r, :) = Pstep(r, :) / sum(Pstep(r, :)); end

res_tbl = []; % [sigma, hmin, events/T, ||x||_2^2, ||u||_2^2]

for hs = hmin_list

    for sgm = sigma_list
        x = [0.2; -0.1]; rk = 1; x_hold = x; u = zeros(m, 1);

        events = 1; evk = zeros(Nstep, 1); evk(1) = 1;
            x2 = 0; u2 = 0;

            for k = 1:Nstep
                rk = find(rand <= cumsum(Pstep(rk, :)), 1);
                Ki = Kv{rk}; Lam = Lambda_eff{rk};
                u = Ki * x_hold;
                x = x + dt * (A{rk} * x + B{rk} * u);

                e = x_hold - x;
                lhs = e.' * Lam * e; rhs = sgm * (x.' * Lam * x);

                if (k - evk(events)) >= hs && lhs >= rhs
                    x_hold = x; events = events + 1; evk(events) = k;
                end

                x2 = x2 + dt * (x.' * x);
                u2 = u2 + dt * (u.' * u);
            end

            ev_rate = events / T;
            res_tbl = [res_tbl; sgm hs ev_rate x2 u2]; %#ok<AGROW>
            fprintf('sigma=%.3g, hmin=%d -> %.1f Hz, ||x||^2=%.3e, ||u||^2=%.3e\n', ...
                sgm, hs, ev_rate, x2, u2);
        end

    end

    % 简单可视化
    figure; hold on; grid on;
    mk = {'o', 's', '^'};

    for i = 1:numel(hmin_list)
        idx = res_tbl(:, 2) == hmin_list(i);
        plot(res_tbl(idx, 1), res_tbl(idx, 3), ['-.' mk{i}], 'LineWidth', 1.5, 'MarkerSize', 7);
    end

    xlabel('\sigma'); ylabel('event rate [Hz]');
    legend(arrayfun(@(x)sprintf('h_{min}=%d', x), hmin_list, 'uni', 0), 'Location', 'best');
    title('事件频率 vs \sigma（不同 h_{min}）');

    save zIE_sweep_sigma.mat res_tbl sigma_list hmin_list -v7
    disp('zIE_sweep_sigma.mat 已生成');
