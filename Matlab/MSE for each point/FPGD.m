function s = FPGD(K, d, A, c, r, s0, L, tol)
% Proximal Gradient Descent

s = s0;
y = s;
t = 1;
delta_f = tol+1;
f_old = DualFunctionEval(K, d, A, c, r, s);
%f_story = [];
%s_story = [s0];
while delta_f > tol
    
    % s_k = prox_(L^-1 q^*)( y_k - 1/L A*A' y_k)
    p0 = y - (1/L)* A * A' * y;
    p = c+L*p0;
    for k = 1:K
        for l = k+1:K
            nb = (K-1/2)*k-k^2/2+l-(K+1);
            
            pkl1 = nb*d+1;
            pkl2 = (nb+1)*d;
            pkl = p(pkl1:pkl2);
            
            rkl = r(nb+1);
            
            if norm(pkl) > rkl               
                p(pkl1:pkl2) = rkl*pkl/norm(pkl);
            end;
            
        end;
    end;
    s_new = (c/L) + p0 - p/L;
    
    % update t
    t_new = (1+sqrt(1+4*t^2))/2;
    
    % update y
    y = s_new + ((t-1)/t_new)*(s_new-s);
    
    % prepare for next iteration
    s = s_new;
    t = t_new;
    
    f_new = DualFunctionEval(K, d, A, c, r, s);
    delta_f = norm(f_new - f_old);
    f_old = f_new;
    %f_story = [f_story f_old];
    %s_story = [s_story s];

end
end 

