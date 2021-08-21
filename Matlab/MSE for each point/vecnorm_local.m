function M = vecnorm_local( A, d )
% vectorwise norm of a matrix, along dimension d
 M = sqrt(sum(A.^2, d));


end

