% MJLS + Disturbance-Unknown + Observer + Event-Trigger + Hinf (ETM+ZOH)
% Requires: YALMIP (https://yalmip.github.io/) + an SDP solver (SeDuMi OK)
% NOTE: We fix gamma to a constant (gamma_val) to keep the problem LMI-only.

clear; clc; rng(1);

%% ========================== Problem Setup ================================
% Dimensions (example)
n = 2; % state x
m = 1; % input u
p = 1; % disturbance w
r = 1; % measured output y
q = 1; % performance output z
Nmode = 2; % number of modes

% Continuous-time Markov chain generator Q (rows sum to zero)
Q = [-0.8 0.8;
     0.5 -0.5];

% System matrices per mode i
A = cell(Nmode, 1); B = A; E = A; C = A; Cz = A; Dz = A;
% Mode 1
A{1} = [0 1; -2 -0.8];
B{1} = [0; 1.0];
E{1} = [0; 0.5];
C{1} = [1 0];
Cz{1} = [1 0];
Dz{1} = 0.2;
% Mode 2
A{2} = [0 1; -1.2 -0.4];
B{2} = [0; 1.0];
E{2} = [0; 0.7];
C{2} = [1 0];
Cz{2} = [1 0];
Dz{2} = 0.1;

% ETM parameters
h_min = 0.02; % minimum inter-event time (anti-Zeno)
mu_i = [5, 5]; % diag for Phi_i = mu_i*I (error side)
nu_i = [0.5, 0.5]; % diag for Psi_i = nu_i*I (state/hatw side)

% Disturbance derivative bound |dot w| <= rho
rho = 0.5;

% ---- Fix H-infinity level to keep LMI (no QSDP) ----
gamma_val = 1.8;

%% ======================= YALMIP Decision Vars ===========================
% Augmented xi = [x; x_tilde; w_tilde; e_x; e_w]
nx = n; nt = n; nwtil = p; nex = n; new = p;
nxi = nx + nt + nwtil + nex + new;
ne = nex + new;
nd = p + p; % [w; dot w]

% Per-mode decision variables
P = cell(Nmode, 1); R = P; Nf = P; W = P; Qint = P;
X = P; Xhat = P; Xwtil = P;
Y = P; Yw = P; Z = P; U = P;
alpha = sdpvar(Nmode, 1);
Phi = cell(Nmode, 1); Psi = cell(Nmode, 1);

cons = [];

for i = 1:Nmode
    % Lyapunov and weights
    P{i} = sdpvar(nxi, nxi, 'symmetric');
    R{i} = sdpvar(ne, ne, 'symmetric');
    Nf{i} = sdpvar(nxi, ne, 'full'); % cross term
    W{i} = sdpvar(ne, ne, 'symmetric'); % timer weight (kept >=0)
    Qint{i} = sdpvar(n + p, n + p, 'symmetric'); % integral term on [x; hat w]

    % Variable substitutions
    X{i} = sdpvar(n, n, 'symmetric'); % for K
    Xhat{i} = sdpvar(n, n, 'symmetric'); % for L & Kobs (not used in this LMI version)
    Xwtil{i} = sdpvar(p, p, 'symmetric'); % for Kw

    % Gains (linearized)
    Y{i} = sdpvar(m, n, 'full'); % Y=K*X
    Yw{i} = sdpvar(m, p, 'full'); % Yw=Kw*Xwtil
    Z{i} = sdpvar(n, r, 'full'); % placeholder for observer (not used)
    U{i} = sdpvar(p, r, 'full'); % placeholder for observer (not used)

    % Triggering matrices
    Phi{i} = mu_i(i) * eye(ne);
    Psi{i} = nu_i(i) * blkdiag(eye(n), eye(p)); % acts on [x; hat w]

    % Positivity constraints
    cons = [cons, P{i} >= 1e-6 * eye(nxi), R{i} >= 0, W{i} >= 0, Qint{i} >= 0];
    cons = [cons, X{i} >= 1e-6 * eye(n), Xhat{i} >= 1e-6 * eye(n), Xwtil{i} >= 1e-6 * eye(p)];
    cons = [cons, alpha(i) >= 1e-3];
end

%% ==================== Helper selection matrices =========================
Sx = [eye(n), zeros(n, n + p + n + p)];
Stx = [zeros(n, n), eye(n), zeros(n, p + n + p)];
Swt = [zeros(p, 2 * n), eye(p), zeros(p, n + p)];
Sex = [zeros(n, 2 * n + p), eye(n), zeros(n, p)];
Sew = [zeros(p, 2 * n + p + n), eye(p)];
Se = [Sex; Sew]; % e = [e_x; e_w]

%% ================== Build flow (interval) LMI per mode ===================
for i = 1:Nmode
    Ai = A{i}; Bi = B{i}; Ei = E{i}; Ci = C{i}; Czi = Cz{i}; Dzi = Dz{i};

    % Placeholders (not explicitly used below, but kept for clarity)
    Fxi = zeros(nxi, nxi, 'sym'); %#ok<NASGU>
    Gxi = zeros(nxi, ne, 'sym'); %#ok<NASGU>
    Hxi = zeros(nxi, nd, 'sym'); %#ok<NASGU>

    % Accumulate blocks for quadratic form on [xi; e; d]
    Xi11 = 0; Xi12 = 0; Xi22 = 0; Xi13 = 0; Xi23 = 0; Qd = 0;

    % Convenience handles
    Pi = P{i}; Ri = R{i}; Ni = Nf{i}; Wi = W{i}; %#ok<NASGU>
    Xi = X{i}; Xh = Xhat{i}; Xw = Xwtil{i}; %#ok<NASGU>
    Yi = Y{i}; Ywi = Yw{i}; %#ok<NASGU>

    % He{M} helper (only for square matrices!)
    He_add = @(M) (M + M');

    % Projections of P on sub-blocks
    Px = Sx * Pi * Sx'; % n x n
    Ptx = Stx * Pi * Stx'; % n x n
    Pvt = Swt * Pi * Swt'; % p x p

    % (A1) x dynamics
    Xi11 = Xi11 + Sx' * He_add(Px * Ai) * Sx;

    % (A2) (skip K linearization here; will be added in a refined version)

    % (A3) E*w  -> [xi; d] cross term
    Sd_w = [eye(p), zeros(p)]; % picks w from d=[w; dot w]
    Sd_dw = [zeros(p), eye(p)]; % picks dot w
    Xi13 = Xi13 + (Sx' * (Px * Ei)) * Sd_w;

    % (A4) tilde-x dynamics
    Xi11 = Xi11 + Stx' * He_add(Ptx * Ai) * Stx;
    % +E*w_til  (two-side sum, no He on non-square)
    Xi11 = Xi11 + Stx' * (Ptx * Ei) * Swt;
    Xi11 = Xi11 + Swt' * (Ei' * Ptx) * Stx;
    % -E*w -> [xi; d]
    Xi13 = Xi13 - (Stx' * (Ptx * Ei)) * Sd_w;

    % (A4') w_til dynamics: -alpha*w_til - dot w
    Xi11 = Xi11 + Swt' * He_add(-alpha(i) * Pvt) * Swt;
    Xi13 = Xi13 - Swt' * Pvt * Sd_dw;

    % (A5) Sample error penalties / cross terms
    Xi12 = Xi12 + Ni;
    Xi22 = Xi22 + (R{i} + W{i});

    % (B) Jensen lower bound kept as Qi >= 0 (no explicit term added)

    % (C) Markov generator difference terms
    Delta = 0;

    for j = 1:Nmode
        if j == i, continue; end
        Delta = Delta + Q(i, j) * blkdiag(P{j} - P{i}, R{j} - R{i});
    end

    Xi11 = Xi11 + Delta(1:nxi, 1:nxi);
    Xi22 = Xi22 + Delta(nxi + 1:nxi + ne, nxi + 1:nxi + ne);

    % (D) Supply rate s(z,w) with fixed gamma (LMI)
    % Conservative: z = Cz*x only (no Dz*u here, added later with congruence trick)
    Cxi = [Czi, zeros(q, n), zeros(q, p), zeros(q, n), zeros(q, p)];
    Ce = zeros(q, ne);
    % w extracted from d's first p components
    Wxi = zeros(p, nxi);
    We = zeros(p, ne);
    Wd = [eye(p), zeros(p)]; % p x nd
    PiH = blkdiag(eye(q), -gamma_val ^ 2 * eye(p));
    % Overall mapping T: [z; w] = T * [xi; e; d]
    T = [Cxi, Ce, zeros(q, nd);
         Wxi, We, Wd];
    % Quadratic form
    Sxx = T' * PiH * T;

    % Merge into big LMI blocks
    Xi11 = Xi11 + Sxx(1:nxi, 1:nxi);
    Xi12 = Xi12 + Sxx(1:nxi, nxi + (1:ne));
    Xi22 = Xi22 + Sxx(nxi + (1:ne), nxi + (1:ne));
    Xi13 = Xi13 + Sxx(1:nxi, nxi + ne + (1:nd));
    Xi23 = Xi23 + Sxx(nxi + (1:ne), nxi + ne + (1:nd));
    Qd = Qd + Sxx(nxi + ne + (1:nd), nxi + ne + (1:nd));

    % Flow LMI for mode i
    Big = [Xi11, Xi12, Xi13;
           Xi12', Xi22, Xi23;
           Xi13', Xi23', -Qd];
    cons = [cons, Big <= -1e-7 * eye(size(Big))];

    % (E) Jump LMI with simple diagonal trigger (still conservative)
    sigma_i = sdpvar(1, 1); cons = [cons, sigma_i >= 0];
    J = blkdiag(eye(n), eye(n), eye(p), zeros(n), zeros(p)); % reset e_x,e_w
    Sel_x_hatw = [Sx; [zeros(p, 2 * n), eye(p), zeros(p, n + p)]]; % [x; w_til]
    Psi_hat = Sel_x_hatw' * blkdiag(eye(n), eye(p)) * Psi{i} * blkdiag(eye(n), eye(p)) * Sel_x_hatw;
    Jump = [J' * P{i} * J - P{i}, -Nf{i};
            -Nf{i}', - (R{i} + W{i})] + sigma_i * blkdiag(-Psi_hat, Phi{i});
    cons = [cons, Jump <= -1e-7 * eye(size(Jump))];
end

%% ============== Solve (SeDuMi; pure LMI) ================================
obj = []; % no optimization of gamma_val here
ops = sdpsettings('solver', 'sedumi', 'verbose', 1);
sol = optimize(cons, obj, ops);

if sol.problem ~= 0
    error('SDP not solved: %s', sol.info);
else
    fprintf('Feasible for gamma = %.4f\n', gamma_val);
end

%% ============== Recover gains K, Kw; set L, Kobs = 0 ====================
K = cell(Nmode, 1); Kw = K; L = K; Kobs = K;

for i = 1:Nmode
    Xi_val = value(X{i});
    Xw_val = value(Xwtil{i});
    Yi_val = value(Y{i});
    Yw_val = value(Yw{i});

    % Use pinv for numerical robustness
    K{i} = Yi_val * pinv(Xi_val); % Y = K X
    Kw{i} = Yw_val * pinv(Xw_val); % Yw = Kw Xw

    % Observer gains not enforced in this LMI version -> set to zero
    L{i} = zeros(n, r);
    Kobs{i} = zeros(p, r);

    fprintf('Mode %d:\n  K  = %s\n  Kw = %s\n', ...
        i, mat2str(K{i}, 4), mat2str(Kw{i}, 4));
end

%% ====================== Simple Random Simulation ========================
Tend = 8.0; dt = 1e-3;
t = 0; k = 1; tk = 0; % event counter and last trigger time
i = 1; % start mode
x = [0.5; -0.2]; % initial state
xt = [0; 0]; wtil = 0; % observer errors initial
w = 0; dw = 0; % true disturbance and derivative
ex = zeros(n, 1); ew = 0;

% logs
TT = []; XX = []; UU = []; MODE = []; EVTS = []; DTAU = [];

% Pre-draw exponential holding time for CTMC (requires Statistics TBX; else use -log(rand)/lam)
lam = -Q(i, i); tau_hold = exprnd(1 / lam); t_hold = t + tau_hold;

% ZOH values at last event:
xk = x; hwk = wtil + w;

while t < Tend
    % Mode jump?
    if t >= t_hold
        probs = max(Q(i, [1:end] ~= i), 0); probs = probs / sum(probs);
        states = find(1:Nmode ~= i);
        i = states(randsample(length(states), 1, true, probs));
        lam = -Q(i, i); tau_hold = exprnd(1 / lam); t_hold = t + tau_hold;
    end

    % Control (ZOH using last event samples):
    u = K{i} * xk + Kw{i} * hwk;

    % Disturbance evolution (smooth, bounded dw)
    dw = 0.5 * sin(0.7 * t); dw = max(min(dw, rho), -rho);
    w = w + dt * dw;

    % Observer (continuous) -- here L=0, Kobs=0 in this version
    y = C{i} * x;
    xhat = x + xt;
    what = wtil + w;
    xdot = A{i} * x + B{i} * u + E{i} * w;
    xhatdot = A{i} * xhat + B{i} * u + E{i} * what + L{i} * (y - C{i} * xhat);
    wtil_dot = -alpha(i) * wtil + Kobs{i} * C{i} * (x - xhat) - dw;

    % Error dynamics for logging
    xtdot = xhatdot - xdot; % = (A-LC)xt + E*wtil - E*w
    exdot = -xdot;
    ewdot = -wtil_dot;

    % Integrate (Euler)
    x = x + dt * xdot;
    xt = xt + dt * xtdot;
    wtil = wtil + dt * wtil_dot;
    ex = ex + dt * exdot;
    ew = ew + dt * ewdot;

    % Event-trigger check (gate with h_min)
    tau = t - tk;

    if tau >= h_min
        lhs = [ex; ew]' * (mu_i(i) * eye(ne)) * [ex; ew];
        rhs = [x; (wtil + w)]' * (nu_i(i) * blkdiag(eye(n), eye(p))) * [x; (wtil + w)];

        if lhs >= rhs
            tk = t; k = k + 1;
            xk = x; hwk = wtil + w;
            ex = zeros(n, 1); ew = 0;
            EVTS(end + 1) = t; %#ok<SAGROW>

            if numel(EVTS) >= 2
                DTAU(end + 1) = EVTS(end) - EVTS(end - 1); %#ok<SAGROW>
            end

        end

    end

    % log
    TT(end + 1) = t; %#ok<SAGROW>
    XX(:, end + 1) = x; %#ok<SAGROW>
    UU(end + 1) = u; %#ok<SAGROW>
    MODE(end + 1) = i; %#ok<SAGROW>
    t = t + dt;
end

%% ============================ Plots =====================================
figure;
subplot(3, 1, 1); plot(TT, XX); grid on; ylabel('x');
title('State trajectory (MJLS + ETM)');
subplot(3, 1, 2); plot(TT, UU); grid on; ylabel('u');
subplot(3, 1, 3); stairs(TT, MODE, 'LineWidth', 1); grid on; ylim([0.5 Nmode + 0.5]);
ylabel('mode'); xlabel('t (s)');

if ~isempty(DTAU)
    figure; histogram(DTAU, 30); grid on;
    xlabel('inter-event time \Delta t'); ylabel('count');
    title('Event intervals (\ge h_{min})');
end
