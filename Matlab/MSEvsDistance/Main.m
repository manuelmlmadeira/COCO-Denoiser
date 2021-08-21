clear all; clc; close all;
rng('default')
n = 1;

Sigma = 10^2 * eye(n);
Sigma_inv = inv(Sigma);
mu = zeros(1,n);
L_real = 1;
L_list = [0.1, 0.55, 1, 1.45, 1.9];

A_eigenvalues = L_real * ones(1,n); %L : - (L/(n-1) * (1-1/k)) : L/k; 
A = diag(A_eigenvalues);
b = 0 * ones(n,1);
x_star = A\b;
f_star = quad_func(A, b, x_star);

% MSE vs Variance
n_reps = 10000;
n_points_considered = 2;
n_L = length(L_list);
distance_list = 0:1:120;
fixed_point = zeros(n,1);

denoised_grads = zeros(n_L, length(distance_list), n, n_points_considered, n_reps);
raw_grads = zeros(n_L, length(distance_list), n, n_points_considered, n_reps);
MSE_denoised_list = zeros(n_points_considered, n_reps, length(distance_list), n_L);
MSE_raw_list = zeros(n_points_considered, n_reps, length(distance_list), n_L);
Constraint_list = zeros(n_reps, length(distance_list), n_L);
for L_index = 1:n_L
    L_considered = L_list(L_index);
    L_index
    for distance_index = 1:length(distance_list)
        distance = distance_list(distance_index);
        variable_point = fixed_point + distance;
        visited_points = [fixed_point, variable_point];
        true_gradients = A * visited_points  - b;
        for rep = 1:n_reps
            noise = mvnrnd(mu, Sigma, n_points_considered).';
            noisy_gradients = true_gradients + noise;
            thetas = COCODenoiser(n, L_considered, Sigma_inv, n_points_considered, noisy_gradients, visited_points);
            deviations_denoised = vecnorm(thetas - true_gradients, 1).^2;
            deviations_raw = vecnorm(noisy_gradients - true_gradients, 1).^2;
            
            denoised_grads(L_index, distance_index, :, :, rep) = thetas - true_gradients;
            raw_grads(L_index, distance_index, :, :, rep) = noisy_gradients - true_gradients;
            MSE_denoised_list(:, rep, distance_index, L_index) = deviations_denoised.';
            MSE_raw_list(:, rep, distance_index, L_index) = deviations_raw.';
            Constraint_list(rep, distance_index, L_index) = isActiveConstraint(L_considered, noisy_gradients, visited_points);
        end      
    end
end

p_active = squeeze(mean(Constraint_list,1));
MSE_singular_denoised = squeeze(mean(MSE_denoised_list, 2));
MSE_total_denoised = squeeze(sum(mean(MSE_denoised_list, 2),1));
Expected_theta = squeeze(vecnorm(mean(denoised_grads, 5), 3));
Expected_raw = squeeze(vecnorm(mean(raw_grads, 5), 3));

save('results2')


