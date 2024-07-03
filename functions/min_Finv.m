function [iSel, zHat] = min_Finv(A, B, k, beta, alpha, iPre, kappa, maxIter)
% Solves the convex relaxed problem with log barrier functions
%	min { tr F^-1 } - kappa sum_{i=1}^m(log(z_i) + log(1-z_i)) }
%	subject to sum(z) = k
%
%   or
%
%	min { tr B F^-1 B^H } - kappa sum_{i=1}^m(log(z_i) + log(1-z_i)) }
%	subject to sum(z) = k
%      
%
%   Input
%   A : measurement matrix
%   B : reconstruction matrix (leave it empty for optimization wrt to x)
%   k : sensor budget 
%   m : number of candidates
%   beta : precision of the noise (scalar)
%   alpha : precision of the parameters (vector length n)
%   iPre : index of pre-selected sensors
%   kappa : parameter for barrier function
%   maxIter : max number of iterations
% 
%   Output
%   iSel : index of selected sensors
%   zHat : convex relaxed solution
%
% Based on S. Joshi and S. Boyd, "Sensor Selection via Convex Optimization"
% www.stanford.edu/~boyd/papers/sensor_selection.html
%
% Backtracking line search parameters
paramA = 0.01;
paramB = 0.5;
tolerance = 1e-8;

if isempty(B)
    xFlag = 1;
else
    xFlag = 0;
end

m = size(A,1);
nPre = length(iPre); 

% Form the equality constraint matrix
if isempty(iPre)
    C = zeros(1, m);
    C(1,:) = 1;
else
    C = zeros(2, m);
    C(1,:) = 1;
    C(2,iPre) = 1;
end
C = sparse(C);

% Initialize candidates
z = ones(m,1) * ( (k-nPre)/(m-nPre) );
z(iPre) = 1-1e-9;

% Newton's method parameters
if isempty(kappa)
    kappa  = 1e-6;
end
if isempty(maxIter)
    maxIter  = 1e3;
end

F = get_F(A, z, beta, alpha);
Finv = inv_pd_matrix(F);

if xFlag
    trFinv = trace(Finv);
else
    trFinv = real(trace(B*Finv*B'));
end

% Ensure that the change in objective due to barriers is small
fz = trFinv - kappa*sum(log(z) + log(1-z));

fprintf('\nIter.  Step_size  Newton_decr.  Objective  trace F^-1\n');
fprintf('   0\t  -- \t     --   %10.3f  %10.3f\n', fz, trFinv);

for i = 1:maxIter

    F = get_F(A, z, beta, alpha);
    Finv = inv_pd_matrix(F);

    if xFlag
        T = A*Finv;
        U = T*A' * beta;
        V = T*T' * beta;
    else
        D = A*(Finv*B');
        V = D*D' * beta;
        U = A*Finv*A' * beta;
    end

    g = -diag(V);
    H = 2*U.*V;
    H = real(H);

    g = g + kappa*(1./(1-z) - 1./z);
    H = H + kappa*diag(1./((1-z).^2) + 1./(z.^2));
   
    Hinv = inv_pd_matrix(H); % most expensive
    Hinvg = Hinv*g;

    Q = C*Hinv*C';
    R = chol(Q);
    Cw = C'*(R\(R'\(C*Hinvg)));
    HinvCw = Hinv*Cw;

    dz = -Hinvg + HinvCw;
    
    iDecz = find(dz < 0);
    iIncz = find(dz > 0);
    s = min([1; 0.99*[-z(iDecz)./dz(iDecz) ; (1-z(iIncz))./dz(iIncz)]]);

    while (1)
        zp = z + s*dz;
        F = get_F(A, zp, beta, alpha);
        Finv = inv_pd_matrix(F);

        if xFlag
            trFinv = trace(Finv);
        else
            trFinv = real(trace(B*Finv*B'));
        end
        
        fzp = trFinv - kappa*sum(log(zp) + log(1-zp));

        if (fzp <= fz + paramA*s*g'*dz)
            break;
        end
        s = paramB*s;
    end
    z = zp; 
    fz = fzp;
    
    fprintf('%4d %10.3f %10.3f %10.3f %10.3f\n', i, s, -g'*dz/2, fz, trFinv);

    if(-g'*dz/2 <= tolerance)
        break;
    end
end

% Thresholded solution
zHat = z;
zSort = sort(zHat);
thres = zSort(m-k);
iSel = (zHat > thres);
zTil = zeros(m,1);
zTil(iSel) = 1;
iSel = find(zTil~=0);

end
