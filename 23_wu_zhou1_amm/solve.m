% Event-Triggered Control Design for semi-MSSs (Wu et al., 2023)
% Full and strict implementation of Theorem 2 for Example 1, Case 1.
% --- VERSION 6 (Final Constraint Logic Fix) ---

clear; clc; yalmip('clear');

%% Step 1: Define Given System Parameters and Constants

% System matrices from Table 1
A{1} = [-0.479908, 5.1546; -3.81625,  14.4723];
A{2} = [-1.60261, 9.1632; -0.5918697, 3.0317];
A{3} = [0.634617, 0.917836; -0.50569, 2.48116];

B{1} = [5.8705212, 0; 0, 15.50107];
B{2} = [10.285129, 0; 0, 2.2282663];
B{3} = [0.7874647, 0; 0, 1.5302844];

C{1} = [0.1, 0; 0.1, 0.1];
C{2} = [0.12, 0.1; 0, 0.1];
C{3} = [0.1, 0; 0, 0.14];

E{1} = [0.1; 0.1]; E{2} = E{1}; E{3} = E{1};
F{1} = [0.1, 0.1]; F{2} = F{1}; F{3} = F{1};

n = size(A{1}, 1); 
m = size(B{1}, 2); 
num_modes = 3;

% Scalar parameters from the paper
alpha = 0.1;
delta = 0.1;
d = 0.05;
sigma = 0.01;
epsilon1 = 0.03;
epsilon2 = 0.01;

% H_infinity performance matrices (Upsilon) for Case 1
Upsilon1 = -eye(n);
Upsilon2 = zeros(n, n); 
Upsilon3 = 5^2 * eye(n);
Upsilon4 = zeros(n);

% Mean transition rates E{lambda(iota)}
lambda_star = [-3.545, 1.7725, 1.7725;
               1.0833, -5.4164, 4.3332;
               3.2,    0.8,    -4];

epsilon = 1e-7;

%% Step 2: Define ALL LMI Decision Variables from Theorem 2
P_bar = sdpvar(n, n, num_modes, 'symmetric');
U_bar = sdpvar(n, n, 'symmetric');
Q1_bar = sdpvar(n, n, 'symmetric');
Q2_bar = sdpvar(n, n, 'full');
Q3_bar = sdpvar(n, n, 'symmetric');
Q_bar = [Q1_bar, Q2_bar; Q2_bar', Q3_bar];

H = sdpvar(n, n, 5, 'full'); 
G = sdpvar(n, n, 4, 'full');

Y_bar = sdpvar(n, n, 'full');
K_bar = sdpvar(m, n, num_modes, 'full');
Lambda_bar = sdpvar(n, n, num_modes, 'symmetric');

rho = sdpvar(1, 1);

%% Step 3: Construct the Complete LMI Constraints in a Loop
Constraints = [];
for i = 1:num_modes
    
    % --- Construct Psi_bar(i) ---
    Psi_bar_11 = lambda_star(i,1)*P_bar(:,:,1) + lambda_star(i,2)*P_bar(:,:,2) + lambda_star(i,3)*P_bar(:,:,3) ...
                 + 2*alpha*P_bar(:,:,i) + (A{i}*Y_bar' + Y_bar*A{i}');
    Psi_bar_11 = Psi_bar_11 - (H(:,:,1)/2 + H(:,:,1)'/2) + (G(:,:,1) + G(:,:,1)');
    Psi_bar_12 = P_bar(:,:,i) + G(:,:,2) - Y_bar + epsilon1*Y_bar*A{i}';
    Psi_bar_13 = H(:,:,1) - H(:,:,2) + G(:,:,3) - G(:,:,1)' + B{i}*K_bar(:,:,i) + epsilon2*Y_bar*A{i}';
    Psi_bar_14 = -H(:,:,3) + G(:,:,4);
    Psi_bar_15 = C{i} - Y_bar*Upsilon2;
    Psi_bar_16 = sqrt(d)*G(:,:,1)';
    Psi_bar_22 = -(epsilon1*Y_bar' + Y_bar*epsilon1);
    Psi_bar_23 = -G(:,:,2)' - epsilon2*Y_bar' + epsilon1*B{i}*K_bar(:,:,i);
    Psi_bar_25 = epsilon1*C{i};
    Psi_bar_26 = sqrt(d)*G(:,:,2)';
    Psi_bar_33 = (H(:,:,2) - H(:,:,1)/2 + H(:,:,2)' - H(:,:,1)'/2) - (G(:,:,3) + G(:,:,3)') ...
                 + (epsilon2*B{i}*K_bar(:,:,i) + K_bar(:,:,i)'*B{i}'*epsilon2) - d*exp(-2*alpha*d)*Q3_bar;
    Psi_bar_34 = -H(:,:,4) - exp(-2*alpha*d)*Q2_bar' - G(:,:,4)';
    Psi_bar_35 = epsilon2*C{i};
    Psi_bar_36 = sqrt(d)*G(:,:,3)';
    Psi_bar_44 = -(H(:,:,5)/2 + H(:,:,5)'/2) - (exp(-2*alpha*d)/d)*Q1_bar;
    Psi_bar_46 = sqrt(d)*G(:,:,4)';
    Psi_bar_55 = -Upsilon3;
    Psi_bar_66 = -exp(-2*alpha*d)*U_bar;
    Psi_bar = [Psi_bar_11, Psi_bar_12, Psi_bar_13, Psi_bar_14, Psi_bar_15, Psi_bar_16, Y_bar*E{i}, rho*F{i}';
               Psi_bar_12', Psi_bar_22, Psi_bar_23, zeros(n),   Psi_bar_25, Psi_bar_26, epsilon1*Y_bar*E{i}, zeros(n,1);
               Psi_bar_13', Psi_bar_23', Psi_bar_33, Psi_bar_34, Psi_bar_35, Psi_bar_36, epsilon2*Y_bar*E{i}, zeros(n,1);
               Psi_bar_14', zeros(n)', Psi_bar_34', Psi_bar_44, zeros(n),   Psi_bar_46, zeros(n,1), zeros(n,1);
               Psi_bar_15', Psi_bar_25', Psi_bar_35', zeros(n)', Psi_bar_55, zeros(n),   zeros(n,1), zeros(n,1);
               Psi_bar_16', Psi_bar_26', Psi_bar_36', Psi_bar_46', zeros(n)', Psi_bar_66, zeros(n,1), zeros(n,1);
               (Y_bar*E{i})', (epsilon1*Y_bar*E{i})', (epsilon2*Y_bar*E{i})', zeros(1,n), zeros(1,n), zeros(1,n), -rho*eye(1), zeros(1,1);
               (rho*F{i}')', zeros(1,n), zeros(1,n), zeros(1,n), zeros(1,n), zeros(1,n), zeros(1,1), -rho*eye(1)];
    Constraints = [Constraints, Psi_bar <= -epsilon*eye(size(Psi_bar))];
    
    % --- Phi_bar(i) construction ---
    Phi_bar_11 = lambda_star(i,1)*P_bar(:,:,1) + lambda_star(i,2)*P_bar(:,:,2) + lambda_star(i,3)*P_bar(:,:,3) ...
                 + 2*alpha*P_bar(:,:,i) + sigma*Lambda_bar(:,:,i) ...
                 + (A{i}*Y_bar' + Y_bar*A{i}') + (B{i}*K_bar(:,:,i) + K_bar(:,:,i)'*B{i}');
    Phi_bar_12 = P_bar(:,:,i) - Y_bar + epsilon1*Y_bar*A{i}' + (epsilon1*B{i}*K_bar(:,:,i))';
    Phi_bar_13 = B{i}*K_bar(:,:,i);
    Phi_bar_14 = C{i} - Y_bar*Upsilon2;
    Phi_bar_22 = -(epsilon1*Y_bar' + Y_bar*epsilon1);
    Phi_bar_23 = epsilon1*B{i}*K_bar(:,:,i);
    Phi_bar_24 = epsilon1*C{i};
    Phi_bar_33 = -Lambda_bar(:,:,i);
    Phi_bar_44 = delta*eye(n) - Upsilon3;
    Phi_bar = [Phi_bar_11, Phi_bar_12, Phi_bar_13, Phi_bar_14, Y_bar*E{i},       rho*F{i}';
               Phi_bar_12', Phi_bar_22, Phi_bar_23, Phi_bar_24, epsilon1*Y_bar*E{i}, zeros(n,1);
               Phi_bar_13', Phi_bar_23', Phi_bar_33, zeros(n),   zeros(n,1),           zeros(n,1);
               Phi_bar_14', Phi_bar_24', zeros(n)', Phi_bar_44, zeros(n,1),           zeros(n,1);
               (Y_bar*E{i})', (epsilon1*Y_bar*E{i})', zeros(1,n), zeros(1,n), -rho*eye(1),        zeros(1,1);
               (rho*F{i}')', zeros(1,n), zeros(1,n), zeros(1,n), zeros(1,1),         -rho*eye(1)];
    Constraints = [Constraints, Phi_bar <= -epsilon*eye(size(Phi_bar))];
    
    % --- FIX IS HERE: Mode-dependent constraints moved inside the loop ---
    Constraints = [Constraints, P_bar(:,:,i) >= epsilon*eye(n)];
    Constraints = [Constraints, Lambda_bar(:,:,i) >= 0];
end

%% Step 4: Add Remaining NON-Mode-dependent Constraints
Constraints = [Constraints, U_bar >= epsilon*eye(n), Q_bar >= epsilon*eye(2*n)];
Constraints = [Constraints, rho >= epsilon];

%% Step 5: Configure Solver and Solve
options = sdpsettings('solver', 'sedumi', 'verbose', 2);
sol = optimize(Constraints, [], options);

%% Step 6: Check Solution and Reconstruct
if sol.problem == 0
    disp('LMIs are feasible. Solver found a solution.');
    
    K_bar_sol = value(K_bar);
    Lambda_bar_sol = value(Lambda_bar);
    Y_bar_sol = value(Y_bar);
    
    for i = 1:num_modes
        K{i} = K_bar_sol(:,:,i) * inv(Y_bar_sol');
        Lambda{i} = inv(Y_bar_sol) * Lambda_bar_sol(:,:,i) * inv(Y_bar_sol');
        
        fprintf('\n--- Results for Mode i = %d ---\n', i);
        disp('Calculated Controller Gain K:');
        disp(K{i});
        disp('Calculated Event-Triggering Matrix Lambda:');
        disp(Lambda{i});
    end
else
    disp('LMIs are not feasible. No solution found.');
    disp(sol.info);
end