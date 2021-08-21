function visited_points = quadratic_COCO(x0, A, b, step_size, tol, nstep_max, mu, sigma, L, npoints)

n = length(x0);
sigma_inv = inv(sigma);
x_t = x0;
visited_points = [x_t];
visited_gradients = [];
true_gradients = [];
steps_counter = 0;

while steps_counter < nstep_max
    K = steps_counter+1;
    noise = mvnrnd(mu,sigma,1).';
    grad = A * x_t - b;
    true_gradients = [true_gradients, grad];
    direction = grad + noise;
    visited_gradients = [visited_gradients, direction];
    consulted_points = visited_points;
    consulted_gradients = visited_gradients;
    theta = COCO_Denoiser( K, n, L, consulted_gradients, consulted_points, tol );
    corrected_direction = theta(:,K);
    x_t = x_t - step_size * corrected_direction;
    visited_points = [visited_points, x_t];
    steps_counter = steps_counter + 1;
end

end

