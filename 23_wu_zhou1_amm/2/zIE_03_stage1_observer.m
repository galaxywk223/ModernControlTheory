%% zIE_03_stage1_observer.m —— 固定 K，求 L1 及对角 Lyapunov 块
clear; clc;
load zIE_sys.mat
load zIE_stage0.mat

yalmip('clear');
ops = sdpsettings('solver', 'sedumi', 'verbose', 1, 'cachesolvers', 1);
ops.sedumi.eps = 1e-8; ops.sedumi.maxiter = 200;

% 对角块
lb = 1e-4; ub = 50;
Qxd = sdpvar(n, 1); Qx = diag(Qxd);
Qwd = sdpvar(q, 1); Qw = diag(Qwd);
Wd = sdpvar(q, 1); W = diag(Wd);
Sd = sdpvar(q, 1); S = diag(Sd);

Z1i = cell(1, N); for i = 1:N, Z1i{i} = sdpvar(n, p, 'full'); end
gammaA = sdpvar(1, 1);

ConsA = [lb <= Qxd <= ub, lb <= Qwd <= ub, lb <= Wd <= ub, lb <= Sd <= ub, ...
             gammaA >= 0, [W eye(q); eye(q) S] >= 0];

for i = 1:N
    Ki = Kv{i}; Ai = A{i}; Bi = B{i}; Ci = C{i}; Di = D{i};
    Th11 = Xv * Ai' + Ai * Xv + Bi * (Ki * Xv) + (Ki * Xv)' * Bi';
    Th12 = Bi * (Ki * Xv); Th22 =- (Th12 + Th12');

    Th14 = Ci; Th24 = -Ci; Th15 = Ci * W; Th25 = -Th15;

    Th33 = Qx * Ai + Ai' * Qx - Z1i{i} * Di - Di' * Z1i{i}';
    Th34 = Qx * Ci; Th35 = zeros(n, q);
    Th44 = -2 * lambda * Qw; Th55 = -2 * lambda * S;

    Th = [Th11 Th12 zeros(n) Th14 Th15;
          Th12' Th22 zeros(n) Th24 Th25;
          zeros(n) zeros(n) Th33 Th34 Th35;
          Th14' Th24' Th34' Th44 zeros(q);
          Th15' Th25' Th35' zeros(q) Th55];
    Th = (Th + Th') / 2;

    ConsA = [ConsA, Th - gammaA * eye(size(Th)) <= 0];
end

solA = optimize(ConsA, gammaA, ops);
assert(solA.problem == 0, 'Stage-1 failed: %s', yalmiperror(solA.problem));
fprintf('Stage-1: gammaA = %.3e\n', value(gammaA));

Qxv = diag(value(Qxd)); Qwv = diag(value(Qwd));
Wv = diag(value(Wd)); Sv = diag(value(Sd));
L1v = cell(1, N); for i = 1:N, L1v{i} = (Qxv \ value(Z1i{i})); end

% 只存数值
save zIE_stage1.mat Qxv Qwv Wv Sv L1v -v7
disp('zIE_stage1.mat 已生成');
