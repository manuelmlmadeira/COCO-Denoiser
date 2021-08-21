% Testing
close all; clear all; clc;

% Initialization
rng(40)
n = 10;

L = 1;
mu = zeros(1,n);
sigma = 10^2*eye(n);
k = 3;
tol = 10^(1);

x0 = 100 * ones(n,1);
A_eigenvalues = L : - (L/(n-1) * (1-1/k)) : L/k;
A = diag(A_eigenvalues);
b = 0 * ones(n,1);
x_star = A\b;
f_star = quad_func(A, b, x_star);
step_size = 1/L;

% First 10 Steps
nstep_max = 10;
n_points_considered = 10;
[visited_points, visited_gradients, s_final] = quadratic_COCO_limited_points(x0, A, b, step_size, tol, nstep_max, mu, sigma, L, n_points_considered);

% Last Step
last_point = visited_points(:,end);
noise = mvnrnd(mu,sigma,1).';
grad = A * last_point - b;
noisy_grad = grad + noise;
G = [noisy_grad, visited_gradients(:,2:end)];
X = [last_point, visited_points(:,2:end)];
%tol = 10^(-5);
n_iter = 1000;
[NoWS_theta, NoWS_dual_story] = COCO_Denoiser_dual_story( n_points_considered, n, L, G, X, n_iter );
[WS_theta, WS_dual_story] = COCO_Denoiser_dual_story_WS( n_points_considered, n, L, G, X, n_iter, s_final );


iter_to_represent = 80;
semilogy(1:iter_to_represent, NoWS_dual_story(1:iter_to_represent) - min(WS_dual_story), 'LineWidth', 2)
hold on 
semilogy(1:iter_to_represent, WS_dual_story(1:iter_to_represent) - min(WS_dual_story), 'LineWidth', 2)
legend('No WS', 'WS')
ylabel('DualFunction(s_i) - DualFunction(s^*)')
xlabel('Iterations')


ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];

fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];

print('-bestfit', fig, 'WarmStarting','-dpdf')