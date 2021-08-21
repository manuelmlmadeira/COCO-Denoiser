function visited_points = quadratic_Adam(x0, A, b, step_size, beta_1, beta_2, epsilon, nstep_max, mu, sigma)

n = length(x0);
x_t = x0;
visited_points = x_t;
steps_counter = 0;
m = zeros(n,1);
v = zeros(n,1);

while steps_counter < nstep_max 
    noise = mvnrnd(mu, sigma,1).';
    grad =  A * x_t - b;
    noisy_grad = grad + noise;
    
    m = beta_1 * m + (1-beta_1) * noisy_grad;
    v = beta_2 * v + (1-beta_2) * noisy_grad.^2;
    m_hat = m/(1-beta_1^(steps_counter+1));
    v_hat = v/(1-beta_2^(steps_counter+1));
    x_t = x_t - step_size * m_hat./(sqrt(v_hat) + epsilon);
    
    visited_points = [visited_points, x_t];
    steps_counter = steps_counter + 1;
end 
end

