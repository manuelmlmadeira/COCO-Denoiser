%% MONTE CARLO EXPERIMENTS
close all; clear all; clc;

% INITIALIZATION

n = 3;

n_reps = 1000;
K_max = 10;
mu = zeros(1,n);
sigma_list = 0:100:1000;
tol = 10^(-6);
l = 10;

L = 1;
k = 3;
A_eigenvalues = L : - (L/(n-1) * (1-1/k)) : L/k;
A = diag(A_eigenvalues);
b = 0 * ones(n,1);
x_star = A\b;
f_star = quad_func(A, b, x_star);

% CC
rng('default')
MSE_list = zeros(K_max, length(sigma_list), n_reps);
for K = 1:K_max
    K
    for sigma_index = 1:length(sigma_list)
        sigma_index
        sigma = sigma_list(sigma_index) * eye(n);
        if sigma == zeros(n)
            sigma_inv = eye(n);
        else
            sigma_inv = inv(sigma);
        end
        for rep = 1:n_reps
            visited_points = l * (rand(n, K)-0.5);
            true_gradients = A * visited_points  - b * ones(1, K);
            noise = mvnrnd(mu, sigma, K).';
            noisy_gradients = true_gradients + noise;
            theta = COCO_Denoiser( K, n, L, noisy_gradients, visited_points, tol);                        
            MSE_list(K, sigma_index, rep) = sum(vecnorm_local(theta - true_gradients, 1).^2);
        end
    end
end
    

save('CC_MSE','sigma_list','MSE_list','K_max', 'n_reps')