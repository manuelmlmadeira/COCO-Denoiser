function [primal_solution, dual_story] = COCO_Denoiser_dual_story_WS( K, d, LP, G, X, n_iter, s_initial)
% for Sigma_inv = 1/sigma^2 * eye(n)

if K == 1
    primal_solution = G;
    
elseif K == 2
    g_1 = G(:,1);
    g_2 = G(:,2);
    x_1 = X(:,1);
    x_2 = X(:,2);
    r = LP * (g_1-g_2).' * (x_1 - x_2);
    
    if norm(g_1 - g_2)^2 <= r
        theta_1 = g_1;
        theta_2 = g_2;
    else 
        theta_1 = (g_1+g_2)/2 + LP/4 * (x_1-x_2) + 1/2 * norm(LP/2 * (x_1 - x_2)) * ((g_1 - g_2) - LP/2 * (x_1-x_2))/norm((g_1 - g_2) - LP/2 * (x_1-x_2));
        theta_2 = (g_1+g_2)/2 - LP/4 * (x_1-x_2) - 1/2 * norm(LP/2 * (x_1 - x_2)) * ((g_1 - g_2) - LP/2 * (x_1-x_2))/norm((g_1 - g_2) - LP/2 * (x_1-x_2));
    end
        
    primal_solution = [theta_1, theta_2];
else
    
    
    
    % Building structures for solution
    A = zeros(K * (K-1)/2 * d, K * d); % matrix which relates the constraints
    c = zeros(K * (K-1)/2 * d, 1); % Center of contraints (balls)
    r = zeros(K * (K-1)/2, 1); % Radius of constraints (balls)
    for k = 1:K
        for l = k+1:K
            nb = (K-1/2)*k-k^2/2+l-(K+1);
            A(nb*d+1:(nb+1)*d, (k-1)*d+1:k*d) = eye(d);
            A(nb*d+1:(nb+1)*d, (l-1)*d+1:l*d) = -eye(d);
            c(nb*d+1:(nb+1)*d, 1) = G(:,k)-(LP/2)*X(:,k)-(G(:,l)-(LP/2)*X(:,l));
            r((nb+1), 1) = (LP/2)*norm(X(:,k)-X(:,l));
        end
    end
    
    % Warm-start
    s0 = s_initial;
    s0(1:K*d) = zeros(K*d,1);

    
    L = (norm(A))^2;

    % Solve dual problem with Fast Proximal Gradient Descent 
    [s_star, dual_story] = FPGD(K, d, A, c, r, s0, L, n_iter);
    alpha_star = - A' * s_star;

    % Obtain Primal Solution
    ALPHA = reshape(alpha_star, [d,K]);
    primal_solution = ALPHA + G;
    
end

end

