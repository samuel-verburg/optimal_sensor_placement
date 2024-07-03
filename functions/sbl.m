function [x, alpha, beta] = sbl(y, A, alpha0, beta0, maxIter, tau)
% Sparse Bayesian Learning
% Given the system 
% y = Ax + e
% where e is iid Gaussian noise, and x is a nx1 sparse vector, it computes 
% the MAP estimate of x, and MP estimates of the noise precision beta and 
% the precision of each parameter alpha. 
% The prior assigned to x is a multivariate normal distribution of zero 
% mean and covariance matrix S = diag(alpha.^-1). 
% It works with complex numbers.
%
% Based on M. E. Tipping, "Sparse Bayesian learning and the relevance vector machine"

[m,n] = size(A);
ATA = A'*A;

% initialize hyperparameters
if isempty(beta0)
    beta = 1;
else
    beta = beta0;
end

if isempty(alpha0)
    alpha = ones(n,1);
else
    alpha = alpha0;
end

if isempty(tau)
    tau = 1E-4;
end

if isempty(maxIter)
    maxIter = 1e4;
end

% parameters for non-informative gamma distributions
a = 1E-3; b = a; c = a; d = a;
ik = 0;

fprintf('\nIter.  residual\n');
fprintf('   0\t   %10.3f\n', 1);

while ik < maxIter
    
    ik = ik + 1;

    % current iteration
    alphaK = alpha;
    
    Sinv = real( beta * ATA + alpha .* eye(n) );
    S = inv_pd_matrix(Sinv);
    Sii = diag(S);
    
    x = beta*S*(A'*y);
    
    gamma = 1-alpha.*Sii;
    alpha = (gamma + 2*a) ./ (abs(x).^2 + 2*b);
%     sigmaSq = abs( (norm(y-A*x)^2 + 2*d) / (m - sum(gamma) + 2*c));
    sigmaSq = ( norm(y-A*x)^2 + sum(gamma)/beta + 2*d ) / ( m + 2*c );
    
    beta = sigmaSq^-1;

    % check convergence
    residual = norm(alpha - alphaK) / norm(alphaK);
    if rem(ik,100) == 0
        fprintf('   %4d %10.3f\n', ik, residual);
    end

    if residual < tau
        break
    end
    
end

if ik == maxIter
    disp('Max. number of iterations reached.')
else
    disp(['Convergence achieved in ' num2str(ik) ' iterations.'])
end
end


