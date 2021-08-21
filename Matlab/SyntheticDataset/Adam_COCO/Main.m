%% Quadratic Function
close all; clear all; clc;

% Initialization
n = 10;
n_reps = 100;

L = 1;
step_size = 10;
beta_1 = 0.5;
beta_2 = 0.999;
epsilon = 10^(-8);
mu = zeros(1,n);
sigma = 10^2*eye(n);
k = 3;
tol = 10^(2);
nstep_max = 32;

x0 = 100 * ones(n,1);
A_eigenvalues = L : - (L/(n-1) * (1-1/k)) : L/k;
A = diag(A_eigenvalues);
b = 0 * ones(n,1);
x_star = A\b;
f_star = quad_func(A, b, x_star);


COCO_visited = zeros(n, nstep_max+1, n_reps);
rng('default')
for rep = 1:n_reps
    rep
    COCO_visited(:,:,rep) = quadratic_Adam_COCO(x0, A, b, step_size, beta_1, beta_2, epsilon, tol, nstep_max, mu, sigma, L);
end
save('COCO')