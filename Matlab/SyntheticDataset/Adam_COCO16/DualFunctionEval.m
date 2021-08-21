function f = DualFunctionEval(K, d, A, c, r, s)

p_conj = 1/2*s'*A*A'*s;
q_conj = 0;
for k =1:K
    for l = k+1:K
        nb = (K-1/2)*k-k^2/2+l-(K+1);
            
        skl1 = nb*d+1;
        skl2 = (nb+1)*d;
        skl = s(skl1:skl2);
        
        ckl1 = nb*d+1;
        ckl2 = (nb+1)*d;
        ckl = c(ckl1:ckl2);
            
        rkl = r(nb+1);
        
        q_conj = q_conj + rkl*norm(skl) - ckl'*skl;
    end
end

f = p_conj + q_conj;

end

