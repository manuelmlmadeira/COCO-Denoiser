function theta = COCODenoiser(n, L, sigma_inv, K, g_K, x_K)

if K == 1
    theta = g_K;
elseif K == 2
    g_1 = g_K(:,1);
    g_2 = g_K(:,2);
    x_1 = x_K(:,1);
    x_2 = x_K(:,2);
    r = L * (g_1-g_2).' * (x_1 - x_2);
    
    if norm(g_1 - g_2)^2 <= r
        theta_1 = g_1;
        theta_2 = g_2;
    else 
        theta_1 = (g_1+g_2)/2 + L/4 * (x_1-x_2) + 1/2 * norm(L/2 * (x_1 - x_2)) * ((g_1 - g_2) - L/2 * (x_1-x_2))/norm((g_1 - g_2) - L/2 * (x_1-x_2));
        theta_2 = (g_1+g_2)/2 - L/4 * (x_1-x_2) - 1/2 * norm(L/2 * (x_1 - x_2)) * ((g_1 - g_2) - L/2 * (x_1-x_2))/norm((g_1 - g_2) - L/2 * (x_1-x_2));
    end
        
    theta = [theta_1, theta_2];
    
else

    cvx_begin quiet
    cvx_solver sedumi

    variables theta(n,K)

    objective = 0;
    for j = 1:K
        g_k = g_K(:,j);
        theta_k = theta(:,j);
        deviation = theta_k - g_k;
        objective = objective + deviation.' * sigma_inv * deviation;
        % quad_form( deviation(j,:).' , sigma_inv )
        % square_pos( norm(deviation(j,:) ))
        % deviation(j,:) * sigma_inv * deviation(j,:).'
    end
    minimize( objective )

    subject to

        for k = 1:(K-1)
            for l = (k+1):K
                x_k = x_K(:,k);
                x_l = x_K(:,l);
                theta_k = theta(:,k);
                theta_l = theta(:,l);
                (theta_k-theta_l).' * (theta_k-theta_l) <= L * (theta_k-theta_l).' * (x_k-x_l);

            end 
        end

    cvx_end
end


end

