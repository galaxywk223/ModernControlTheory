%% zIE_04_stageB_full_trigger.m —— 固定 K,L1，对称满矩阵触发权 Gx,Gh
clear; clc;
load zIE_sys.mat
load zIE_stage0.mat
load zIE_stage1.mat

yalmip('clear');
ops = sdpsettings('solver', 'sedumi', 'verbose', 1, 'cachesolvers', 1);
ops.sedumi.eps = 1e-8; ops.sedumi.maxiter = 200;

Gx = cell(1, N); Gh = cell(1, N);
for i = 1:N, Gx{i} = sdpvar(n, n, 'symmetric'); Gh{i} = sdpvar(q, q, 'symmetric'); end

gmin = 1e-3; gmax = 50; hmin = 1e-3; hmax = 50;
gammaB = sdpvar(1, 1); ConsB = [gammaB >= 0];

for i = 1:N
    Ai = A{i}; Bi = B{i}; Ci = C{i}; Di = D{i}; Ki = Kv{i};
    Th11 = Xv * Ai' + Ai * Xv + Bi * (Ki * Xv) + (Ki * Xv)' * Bi';
    Th12 = Bi * (Ki * Xv); Th22 =- (Th12 + Th12');

    Th14 = Ci; Th24 = -Ci; Th15 = Ci * Wv; Th25 = -Th15;

    Th33 = Qxv * Ai + Ai' * Qxv - (Qxv * L1v{i}) * Di - Di' * (Qxv * L1v{i})';
    Th34 = Qxv * Ci; Th35 = zeros(n, q);
    Th44 = -2 * lambda * Qwv; Th55 = -2 * lambda * Sv;

    Th = [Th11 Th12 zeros(n) Th14 Th15;
          Th12' Th22 zeros(n) Th24 Th25;
          zeros(n) zeros(n) Th33 Th34 Th35;
          Th14' Th24' Th34' Th44 zeros(q);
          Th15' Th25' Th35' zeros(q) Th55];
    Th = (Th + Th') / 2;

    M = blkdiag(-sigma * Gx{i}, Gx{i}, zeros(n), zeros(q), -delta * Gh{i});
    ConsB = [ConsB, Th + M - gammaB * eye(size(Th)) <= 0, ...
                 Gx{i} >= gmin * eye(n), gmax * eye(n) - Gx{i} >= 0, ...
                 Gh{i} >= hmin * eye(q), hmax * eye(q) - Gh{i} >= 0];
end

w = 1e-3; pen = 0; for i = 1:N, pen = pen + trace(Gx{i}) + trace(Gh{i}); end
solB = optimize(ConsB, gammaB + w * pen, ops);
assert(solB.problem == 0, 'Stage-B failed: %s', yalmiperror(solB.problem));
fprintf('Stage-B(full): gammaB = %.3e\n', value(gammaB));

Gxv = cell(1, N); Ghv = cell(1, N); Lambda_eff = cell(1, N);

for i = 1:N
    Gxv{i} = value(Gx{i}); Ghv{i} = value(Gh{i});
    Lambda_eff{i} = (Xv' \ Gxv{i}) / Xv;
end

% 只存数值
save zIE_stageB.mat Gxv Ghv Lambda_eff -v7
disp('zIE_stageB.mat 已生成');

% 打印闭环特征值
fprintf('\n==== Closed-loop eigenvalues per mode (A+B*K) ====\n');

for i = 1:N
    ev = eig(A{i} + B{i} * Kv{i});
    fprintf('Mode %d: ', i); fprintf('%.3f%+.3fi  ', [real(ev) imag(ev)]'); fprintf('\n');
end
