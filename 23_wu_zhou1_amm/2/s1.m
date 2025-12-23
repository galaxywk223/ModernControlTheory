% --- SeDuMi path (按需改) ---
addpath(genpath('D:\Software\sedumi\sedumi'));

% --- Clean & solver ---
yalmip('clear'); rehash; clc;
ops = sdpsettings('solver', 'sedumi', 'verbose', 1, 'cachesolvers', 1);
ops.sedumi.eps = 1e-8;
ops.sedumi.maxiter = 200;

%% ===== System (DC motor; from paper) =====
n = 2; q = 2; m = 2; p = 2; N = 3;

A = cell(1, N); B = cell(1, N); C = cell(1, N); D = cell(1, N);
A{1} = [-0.479908 5.1546; -3.81625 14.4723];
B{1} = [5.8705212 0; 0 15.50107];
C{1} = [0.1 0; 0.1 0.1]; D{1} = eye(2);

A{2} = [-1.60261 9.1632; -0.5918697 3.0317];
B{2} = [10.285129 0; 0 2.2282663];
C{2} = [0.12 0.1; 0 0.1]; D{2} = eye(2);

A{3} = [0.634617 0.917836; -0.50569 2.48116];
B{3} = [0.7874647 0; 0 1.5302844];
C{3} = [0.1 0; 0 0.14]; D{3} = eye(2);

% Parameters
sigma = 0.01; % trigger weight on x
delta = 0.02; % trigger weight on hat{w}
lambda = 0.40; % disturbance model rate

%% ========= Stage-0: P-domain stabilization (solve K) =========
lbP = 1e-4; ubP = 1e2;
Pd = sdpvar(n, 1); P = diag(Pd);
Yi = cell(1, N); % Yi = K_i * P
for i = 1:N, Yi{i} = sdpvar(m, n, 'full'); end

alpha = sdpvar(1, 1); % >=0
Cons0 = [lbP <= Pd <= ubP, alpha >= 0];

% cap ||Yi||_2 via Schur
betaY = 5;

for i = 1:N
    Ai = A{i}; Bi = B{i};
    S = Ai * P + Bi * Yi{i}; % Acl*P
    % CORRECT sign: (S+S') - alpha*I <= 0  (minimize alpha)
    Cons0 = [Cons0, (S + S') - alpha * eye(n) <= 0, ...
                 [betaY * eye(m) Yi{i}; Yi{i}' betaY * eye(n)] >= 0];
end

sol0 = optimize(Cons0, alpha, ops);

if sol0.problem ~= 0
    fprintf('Stage-0 status: %d (%s)\n', sol0.problem, yalmiperror(sol0.problem));
    error('Stage-0 failed.');
end

fprintf('Stage-0: alpha = %.3e\n', value(alpha));

Pv = value(P);
Xv = (Pv \ eye(n)); % X = P^{-1} (stable linear solve)
Kv = cell(1, N);

for i = 1:N
    Kv{i} = value(Yi{i}) / Pv; % K_i
end

%% ========= Stage-1: fix K, solve L1 and observer Lyapunov blocks =========
lb = 1e-4; ub = 50;
Qxd = sdpvar(n, 1); Qx = diag(Qxd);
Qwd = sdpvar(q, 1); Qw = diag(Qwd);
Wd = sdpvar(q, 1); W = diag(Wd);
Sd = sdpvar(q, 1); S = diag(Sd);

Z1i = cell(1, N); % Z1_i = Qx * L1_i
for i = 1:N, Z1i{i} = sdpvar(n, p, 'full'); end

gammaA = sdpvar(1, 1);
ConsA = [lb <= Qxd <= ub, lb <= Qwd <= ub, lb <= Wd <= ub, lb <= Sd <= ub, ...
             gammaA >= 0, [W eye(q); eye(q) S] >= 0];

for i = 1:N
    Ai = A{i}; Bi = B{i}; Ci = C{i}; Di = D{i};

    % fixed K
    Th11 = Xv * Ai' + Ai * Xv + Bi * (Kv{i} * Xv) + (Kv{i} * Xv)' * Bi';
    Th12 = Bi * (Kv{i} * Xv);
    Th22 =- (Bi * (Kv{i} * Xv) + (Kv{i} * Xv)' * Bi');

    % observer blocks (F=0, L2=0 here)
    Th14 = Ci; Th24 = -Ci;
    Th15 = Ci * W; Th25 = -Th15;

    Th33 = Qx * Ai + Ai' * Qx - Z1i{i} * Di - Di' * Z1i{i}';
    Th34 = Qx * Ci;
    Th35 = zeros(n, q);
    Th44 = -2 * lambda * Qw;
    Th55 = -2 * lambda * S;

    Th = [Th11 Th12 zeros(n) Th14 Th15;
          Th12' Th22 zeros(n) Th24 Th25;
          zeros(n) zeros(n) Th33 Th34 Th35;
          Th14' Th24' Th34' Th44 zeros(q);
          Th15' Th25' Th35' zeros(q) Th55];
    Th = (Th + Th') / 2;

    % CORRECT sign: Th - gammaA*I <= 0 (minimize gammaA)
    ConsA = [ConsA, Th - gammaA * eye(size(Th)) <= 0];
end

solA = optimize(ConsA, gammaA, ops);

if solA.problem ~= 0
    fprintf('Stage-1 status: %d (%s)\n', solA.problem, yalmiperror(solA.problem));
    error('Stage-1 failed.');
end

fprintf('Stage-1: gammaA = %.3e\n', value(gammaA));

Qxv = value(Qx); Qwv = value(Qw); Wv = value(W); Sv = value(S);
L1v = cell(1, N);

for i = 1:N
    L1v{i} = Qxv \ value(Z1i{i}); % L1_i
end

Fv = cell(1, N); L2v = cell(1, N);
for i = 1:N, Fv{i} = zeros(m, q); L2v{i} = zeros(q, p); end

%% ========= Stage-B (full PSD): fix K,L1, solve full-matrix Gx,Gh =========
% 从对角 -> 全矩阵；用 LMI 约束 gmin*I <= Gx <= gmax*I（Gh 同理）
Gx = cell(1, N); Gh = cell(1, N);

for i = 1:N
    Gx{i} = sdpvar(n, n, 'symmetric');
    Gh{i} = sdpvar(q, q, 'symmetric');
end

gmin = 1e-3; gmax = 50; % 与上一小步一致的上下界
hmin = 1e-3; hmax = 50;

gammaB = sdpvar(1, 1);
ConsB = [gammaB >= 0];

for i = 1:N
    Ai = A{i}; Bi = B{i}; Ci = C{i}; Di = D{i};

    Th11 = Xv * Ai' + Ai * Xv + Bi * (Kv{i} * Xv) + (Kv{i} * Xv)' * Bi';
    Th12 = Bi * (Kv{i} * Xv);
    Th22 =- (Bi * (Kv{i} * Xv) + (Kv{i} * Xv)' * Bi');

    Th14 = Ci; Th24 = -Ci;
    Th15 = Ci * Wv; Th25 = -Th15;

    Th33 = Qxv * Ai + Ai' * Qxv - (Qxv * L1v{i}) * Di - Di' * (Qxv * L1v{i})';
    Th34 = Qxv * Ci;
    Th35 = zeros(n, q);
    Th44 = -2 * lambda * Qwv;
    Th55 = -2 * lambda * Sv;

    Th = [Th11 Th12 zeros(n) Th14 Th15;
          Th12' Th22 zeros(n) Th24 Th25;
          zeros(n) zeros(n) Th33 Th34 Th35;
          Th14' Th24' Th34' Th44 zeros(q);
          Th15' Th25' Th35' zeros(q) Th55];
    Th = (Th + Th') / 2;

    % S-程序项：diag(-σ Gx, Gx, 0, 0, -δ Gh)
    M = blkdiag(-sigma * Gx{i}, Gx{i}, zeros(n), zeros(q), -delta * Gh{i});

    % LMI 边界：gmin*I <= Gx <= gmax*I，hmin*I <= Gh <= hmax*I
    ConsB = [ConsB, Th + M - gammaB * eye(size(Th)) <= 0, ...
                 Gx{i} >= gmin * eye(n), gmax * eye(n) - Gx{i} >= 0, ...
                 Gh{i} >= hmin * eye(q), hmax * eye(q) - Gh{i} >= 0];
end

% 目标：最小化 gammaB，同时温和惩罚 trace(Gx)+trace(Gh) 防止过大
w = 1e-3;
pen = 0;

for i = 1:N
    pen = pen + trace(Gx{i}) + trace(Gh{i});
end

solB = optimize(ConsB, gammaB + w * pen, ops);

if solB.problem ~= 0
    fprintf('Stage-B(full) status: %d (%s)\n', solB.problem, yalmiperror(solB.problem));
    error('Stage-B(full) failed.');
end

fprintf('Stage-B(full): gammaB = %.3e\n', value(gammaB));

% 等效触发矩阵：Lambda_eff = X^{-T} Gx X^{-1}
Lambda_eff = cell(1, N);

for i = 1:N
    Gxv = value(Gx{i});
    Lambda_eff{i} = (Xv' \ Gxv) / Xv;
end

disp('==== Full-matrix Trigger matrices (effective) ====');

for i = 1:N
    fprintf('Lambda_eff{%d} = \n', i);
    disp(Lambda_eff{i});
end

% === Quick check: closed-loop eigenvalues (per mode) ===
fprintf('\n==== Closed-loop eigenvalues per mode (A+B*K) ====\n');

for i = 1:N
    Acl = A{i} + B{i} * Kv{i};
    ev = eig(Acl);
    fprintf('Mode %d eig(A+B*K): ', i);
    fprintf('%.3f%+.3fi  ', [real(ev) imag(ev)]');
    fprintf('\n');
end

%% ===== Minimal simulation: ET feedback without hat{w} term =====
rng(1);
T = 5; % total time [s]
dt = 1e-3; % step
Nstep = round(T / dt);
sigma_sim = 0.01; % use same sigma
delta_sim = 0.0; % <-- disable hat{w} term in trigger (we'll add it next step)
hmin_steps = 3; % min inter-event steps (anti-Zeno)

% Markov switching (constant rate -> discrete jump prob per step)
Pi = [-1.0 0.6 0.4;
      0.5 -1.2 0.7;
      0.4 0.5 -0.9]; % example generator (rows sum to 0)
Pstep = eye(3) + Pi * dt; % first-order approx, small dt OK
Pstep = max(Pstep, 0); % clip negatives
for r = 1:3, Pstep(r, :) = Pstep(r, :) / sum(Pstep(r, :)); end

% Preload Lambda_eff from stage-B(full)
LamEff = Lambda_eff; % {1..3}

% init
x = [0.2; -0.1]; % initial state
rk = 1; % initial mode index
u = zeros(m, 1);
tk_last = 0; % last event time
x_hold = x; % x(t_k)

events = 1; % event counter
    ev_idx = zeros(Nstep, 1); ev_idx(1) = 1;

    Xlog = zeros(n, Nstep); Ulog = zeros(m, Nstep);
    Rlog = zeros(1, Nstep); Tlog = (0:Nstep - 1) * dt;

    for k = 1:Nstep
        t = Tlog(k);
        % 1) Markov jump
        pr = Pstep(rk, :);
        rk = find(rand <= cumsum(pr), 1, 'first');

        Ai = A{rk}; Bi = B{rk}; Ci = C{rk}; %#ok<NASGU> (Ci未用)
        Ki = Kv{rk}; Lam = LamEff{rk};

        % 2) control (ZOH on x_hold)
        u = Ki * x_hold;

        % 3) plant (no explicit disturbance yet)
        xdot = Ai * x + Bi * u;
        x = x + dt * xdot;

        % 4) event-trigger check (with min dwell)
        e = x_hold - x;
        lhs = e.' * Lam * e;
        rhs = sigma_sim * (x.' * Lam * x) + delta_sim * 0; % next step we'll add hat{w} term

        if (k - ev_idx(events)) >= hmin_steps && lhs >= rhs
            % trigger
            x_hold = x;

            events = events + 1;
                ev_idx(events) = k;
            end

            % log
            Xlog(:, k) = x; Ulog(:, k) = u; Rlog(k) = rk;
        end

        ev_idx = ev_idx(1:events);

        fprintf('\n[SIM] events = %d over %.2fs (avg %.1f Hz)\n', events, T, events / T);

        % quick plots (MATLAB)
        figure; plot(Tlog, Xlog); xlabel('t [s]'); ylabel('x'); title('States');
        figure; plot(Tlog, Ulog); xlabel('t [s]'); ylabel('u'); title('Control');
        figure; stairs(Tlog, Rlog); xlabel('t [s]'); ylabel('mode'); title('Mode r(t)');
        figure; stem(Tlog(ev_idx), ones(numel(ev_idx), 1)); xlabel('t [s]'); title('Event times');
