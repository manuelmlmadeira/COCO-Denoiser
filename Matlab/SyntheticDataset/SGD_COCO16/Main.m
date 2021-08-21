%% Quadratic Function
close all; clear all; clc;

% Initialization
n = 10;
n_reps = 100;

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
nstep_max = 32;

npoints = 16;
COCO_visited = zeros(n, nstep_max+1, n_reps);
rng('default')
for rep = 1:n_reps
    rep
    COCO_visited(:,:,rep) = quadratic_COCO_limited_points(x0, A, b, step_size, tol, nstep_max, mu, sigma, L, npoints);
end
save('COCO16')