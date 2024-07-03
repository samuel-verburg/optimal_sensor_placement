function [varX, varBx, Finv] = get_crb(iSel, A, B, beta, alpha)
% Compute Cramer-Rao bound

F = get_F(A(iSel,:), 1, beta, alpha);
Finv = inv_pd_matrix(F);

varX = trace(Finv);
varBx = real(trace(B*Finv*B'));

end