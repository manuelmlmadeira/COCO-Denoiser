function visited_points = quadratic_SGD (x0, A, b, step_size, nstep_max, mu, sigma)

x_t = x0;
visited_points = x_t;
steps_counter = 0;
while steps_counter < nstep_max 
    noise = mvnrnd(mu,sigma,1).';
    grad =  A * x_t - b;
    direction = grad + noise;
    x_t = x_t - step_size * direction;
    visited_points = [visited_points, x_t];
    steps_counter = steps_counter + 1;
   
end 
end

