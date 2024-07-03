function [varX, varBx, Finv] = get_crb_z(z, A, B, beta, alpha)
% Compute Cramer-Rao bound with non-integer z

F = get_F(A, z, beta, alpha);
Finv = inv_pd_matrix(F);

varX = trace(Finv);
varBx = real(trace(B*Finv*B'));

end