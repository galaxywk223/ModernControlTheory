%% zIE_02_stage0_K.m  —— P域稳定化：求各模态 K_i
clear; clc;
load zIE_sys.mat

yalmip('clear');
ops = sdpsettings('solver', 'sedumi', 'verbose', 1, 'cachesolvers', 1);
ops.sedumi.eps = 1e-8; ops.sedumi.maxiter = 200;

% 变量（对角 P，Y_i = K_i P）
lbP = 1e-4; ubP = 1e2;
Pd = sdpvar(n, 1); P = diag(Pd);
Yi = cell(1, N); for i = 1:N, Yi{i} = sdpvar(m, n, 'full'); end
alpha = sdpvar(1, 1); betaY = 5;

Cons0 = [lbP <= Pd <= ubP, alpha >= 0];

for i = 1:N
    S = A{i} * P + B{i} * Yi{i}; % Acl*P
    Cons0 = [Cons0, (S + S') - alpha * eye(n) <= 0, ...
                 [betaY * eye(m) Yi{i}; Yi{i}' betaY * eye(n)] >= 0];
end

sol0 = optimize(Cons0, alpha, ops);
assert(sol0.problem == 0, 'Stage-0 failed: %s', yalmiperror(sol0.problem));
fprintf('Stage-0: alpha = %.3e\n', value(alpha));

% 数值结果
Pv = diag(value(Pd)); Xv = Pv \ eye(n);
Kv = cell(1, N); for i = 1:N, Kv{i} = value(Yi{i}) / Pv; end

% 只存数值
save zIE_stage0.mat Pv Xv Kv -v7
disp('zIE_stage0.mat 已生成');
