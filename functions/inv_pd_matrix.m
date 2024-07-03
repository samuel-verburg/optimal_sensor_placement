function Ainv = inv_pd_matrix(A)
% Calculates the inverse of a symmetric positive definte matrix explicitly
% using the Cholesky factorization

n = size(A,1);
I = eye(n);

R = chol(A);
Rinv = R \ I;
Ainv = Rinv * Rinv';

Ainv = real_diag(Ainv);

end