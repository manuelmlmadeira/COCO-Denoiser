clear all; clc; close all;

rng(1)
% MSE vs Variance
n_reps = 1000;
n_points_considered = 8;
n_configurations = 6;
n = 3;
tol  = 10^-6;

sigma = 10;
Sigma = sigma^2 * eye(n);
Sigma_inv = inv(Sigma);
mu = zeros(1,n);
L = 1;
k = 3;

A_eigenvalues = L : - (L/(n-1) * (1-1/k)) : L/k; 
A = diag(A_eigenvalues);
b = 0 * ones(n,1);
x_star = A\b;
f_star = quad_func(A, b, x_star);


MSE_denoised_list = zeros(n_configurations, n_points_considered, n_reps);
MSE_raw_list = zeros(n_configurations, n_points_considered, n_reps);
visited_points_history = zeros(n, n_points_considered, n_configurations);
for config_index = 1:n_configurations
    visited_points = 2000 * (rand(n, n_points_considered)-0.5);
    visited_points_history(:, :, config_index) = visited_points;
    true_gradients = A * visited_points  - b;
    for rep = 1:n_reps
        rep
        noise = mvnrnd(mu, Sigma, n_points_considered).';
        noisy_gradients = true_gradients + noise;
        thetas = COCO_Denoiser( n_points_considered, n, L, noisy_gradients, visited_points, tol );
        deviations_denoised = vecnorm_local(thetas - true_gradients, 1).^2;
        deviations_raw = vecnorm_local(noisy_gradients - true_gradients, 1).^2;

        MSE_denoised_list(config_index, :, rep) = deviations_denoised.';
        MSE_raw_list(config_index, :, rep) = deviations_raw.';
   
    end 
end

  
MSE_singular_denoised = mean(MSE_denoised_list, 3);
STD_singular_denoised = std(MSE_denoised_list, 0, 3)/sqrt(n_reps);
MSE_singular_raw = mean(MSE_raw_list, 3);
STD_singular_raw = std(MSE_raw_list, 0, 3)/sqrt(n_reps);

%save('results3')



