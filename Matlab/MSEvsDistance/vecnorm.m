function M = vecnorm( A, d )
% vectorwise norm of a matrix, along dimension d
 M = sqrt(sum(A.^2, d));


end

