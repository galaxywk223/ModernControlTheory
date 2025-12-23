function animate_cartpole(t, X, opt)
% 简洁小车-倒立摆动画（直立平衡附近，theta=0 为竖直向上）
% t: 时间向量（单调递增）
% X: 状态矩阵 [x, xdot, theta, thetadot]
% opt: 结构体参数（见调用处）

if nargin < 3, opt = struct; end
if ~isfield(opt,'dt_visual'),     opt.dt_visual   = 0.01; end
if ~isfield(opt,'track_width'),   opt.track_width = 0.3;  end
if ~isfield(opt,'track_height'),  opt.track_height= 0.15; end
if ~isfield(opt,'rod_len'),       opt.rod_len     = 0.5;  end
if ~isfield(opt,'x_lim'),         opt.x_lim       = 1.5;  end
if ~isfield(opt,'y_base'),        opt.y_base      = -0.2; end

% 为平滑动画做时间插值
t_vis      = t(1):opt.dt_visual:t(end);
x_vis     = interp1(t, X(:,1), t_vis, 'pchip');
theta_vis = interp1(t, X(:,3), t_vis, 'pchip');

% 图形初始化
figure('Color','w','Name','Cart-Pole Animation'); clf;
ax = axes('XLim',[-opt.x_lim opt.x_lim], ...
          'YLim',[opt.y_base-0.1 opt.rod_len+0.4], ...
          'DataAspectRatio',[1 1 1]); 
hold(ax,'on'); grid(ax,'on');
xlabel('x [m]'); ylabel('y [m]'); title('Cart-Pole (upright, linearized)');

% 轨道
plot(ax, [-opt.x_lim opt.x_lim], [opt.y_base opt.y_base], 'k-', 'LineWidth',1);

% 小车与杆的图元
cart_w = opt.track_width;
cart_h = opt.track_height;

cart = rectangle(ax, 'Position',[x_vis(1)-cart_w/2, opt.y_base, cart_w, cart_h], ...
                 'Curvature',0.1, 'FaceColor',[0.7 0.7 0.7], 'EdgeColor','k');
pin  = plot(ax, x_vis(1), opt.y_base+cart_h, 'ko', 'MarkerSize',6, 'MarkerFaceColor','k');
rod  = line(ax, [x_vis(1), x_vis(1)], [opt.y_base+cart_h, opt.y_base+cart_h+opt.rod_len], ...
            'LineWidth',2);
bob  = plot(ax, x_vis(1), opt.y_base+cart_h+opt.rod_len, 'o', 'MarkerSize',8, 'MarkerFaceColor','r');

% 时间文本
time_txt = text(ax, -opt.x_lim+0.05, opt.y_base+0.05, sprintf('t = %.2f s', t_vis(1)));

% 动画循环
for k = 1:numel(t_vis)
    xk = x_vis(k);
    th = theta_vis(k); % theta=0 竖直向上，逆时针为正
    
    % 枢轴点坐标
    px = xk; 
    py = opt.y_base + cart_h;
    % 杆端点坐标（相对竖直向上）
    ex = px + opt.rod_len * sin(th);
    ey = py + opt.rod_len * cos(th);
    
    % 更新图元
    cart.Position = [xk - cart_w/2, opt.y_base, cart_w, cart_h];
    set(pin, 'XData', px, 'YData', py);
    set(rod, 'XData', [px ex], 'YData', [py ey]);
    set(bob, 'XData', ex, 'YData', ey);
    set(time_txt, 'String', sprintf('t = %.2f s', t_vis(k)));
    
    drawnow;
end
end
