function dx = cartpole_lin_closedloop(~, x, p)
% x = [x; xdot; theta; thetadot]
% 闭环：u = -Kx  -> xdot = (A - B*K) x
Acl = p.A - p.B * p.K;
dx  = Acl * x;
end
