function output = isActiveConstraint(L, g_K, x_K)

    if size(g_K,2) == 1
        output = 0;
    elseif size(g_K,2) == 2
        g_1 = g_K(:,1);
        g_2 = g_K(:,2);
        x_1 = x_K(:,1);
        x_2 = x_K(:,2);
        r = L * (g_1-g_2).' * (x_1 - x_2);
        if norm(g_1 - g_2)^2 <= r
            output = 0;
        else 
            output = 1;
        end
    else
        disp('wrong number of gradients fed')
    
end

