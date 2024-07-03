function F = get_F(A,z,beta,alpha)
% Calculate the Bayesian FIM (linear model)
%   Input 
%   A : matrix of candidates
%   z : selection vector
%   beta : precision of the noise 
%   alpha : precision of the parameters 
%
%   Output
%   F : Bayesian Fisher Information Matrix

n = size(A,2);
F = beta*(A'*(z.*A)) + alpha.*eye(n);
F = real_diag(F);

end