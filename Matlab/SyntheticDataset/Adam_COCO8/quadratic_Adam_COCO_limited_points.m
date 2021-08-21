function visited_points = quadratic_Adam_COCO_limited_points(x0, A, b, step_size, beta_1, beta_2, epsilon, tol, nstep_max, mu, sigma, L, npoints)

n = length(x0);
sigma_inv = inv(sigma);
x_t = x0;
visited_points = [x_t];
visited_gradients = [];
true_gradients = [];
steps_counter = 0;
m = zeros(n,1);
v = zeros(n,1);

while steps_counter < nstep_max
    if steps_counter+1 > npoints
        K = npoints;
    else
        K = steps_counter+1;
    end
    noise = mvnrnd(mu,sigma,1).';
    grad = A * x_t - b;
    true_gradients = [true_gradients, grad];
    direction = grad + noise;
    visited_gradients = [visited_gradients, direction];
    if steps_counter+1 > npoints
        consulted_points = visited_points(:,(end-K+1):end);
        consulted_gradients = visited_gradients(:,(end-K+1):end);
    else 
        consulted_points = visited_points;
        consulted_gradients = visited_gradients;
    end
    theta = COCO_Denoiser( K, n, L, consulted_gradients, consulted_points, tol );
    corrected_noisy_grad = theta(:,K);
    
    m = beta_1 * m + (1-beta_1) * corrected_noisy_grad;
    v = beta_2 * v + (1-beta_2) * corrected_noisy_grad.^2;
    m_hat = m/(1-beta_1^(steps_counter+1));
    v_hat = v/(1-beta_2^(steps_counter+1));
    x_t = x_t - step_size * m_hat./(sqrt(v_hat) + epsilon);
    
    visited_points = [visited_points, x_t];
    steps_counter = steps_counter + 1;
end

end

