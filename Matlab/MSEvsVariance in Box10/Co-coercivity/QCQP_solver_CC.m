function theta = QCQP_solver_CC(n, L, sigma_inv, K, g_K, x_K)
 
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

