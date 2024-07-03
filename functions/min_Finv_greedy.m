function iSel = min_Finv_greedy(A, B, k, beta, alpha, iPre)
% Greedy search for 
%	min { tr F^-1 }
%	subject to sum(z) = k
%              z in {0,1}
%
%   or
%
%	min { tr B F^-1 B^H }
%	subject to sum(z) = k
%              z in {0,1}
%
% The sensors are selected one by one. The ith sensor is selected by
% computing the objective function f using the rows of A corresponding to 
% the already selected sensors and one of the remaining rows. A sweep 
% through all the remaning rows is done, and the sensor that minimizes f is
% kept. The process is repeated until k sensors are selected. 
% 
% This process results in a local minimum for the optimization problem. 
% A low-rank update of the F^-1 is performed for efficiency.
%
%   Input
%   A : measurement matrix
%   B : reconstruction matrix (leave it empty for optimization wrt to x)
%   k : sensor budget 
%   m : number of candidates
%   beta : precision of the noise (scalar)
%   alpha : precision of the parameters (vector length n)
%   iPre : index of pre-selected sensors
% 
%   Output
%   iSel : index of selected sensors

if isempty(B)
    xFlag = 1;
else
    xFlag = 0;
end

m = size(A,1);

if ~isempty(iPre)
    sel = iPre;
else
    sel = [];
end

ind = 1:m;

while length(sel) < k
    
    rem = setdiff(ind, sel); 
    
    in = rem(1);
    z = zeros(m,1);
    z([sel, in]) = 1;
    
    F = get_F(A, z, beta, alpha);
    Finv = inv_pd_matrix(F);

    if xFlag
        trFinv = trace(Finv);
    else
        trFinv = real(trace(B*Finv*B'));
    end

    fz = trFinv;
    
    for iSweep = 1:length(rem)-1
        j = rem(iSweep);
        l = rem(iSweep+1);
        
        aj = A(j,:);
        al = A(l,:);
        U = beta*[-aj' al'];
        Vt = [aj; al];
        S = eye(2) + Vt * Finv * U;
        Finv = Finv - Finv*U*inv(S)*Vt*Finv; % Woodbury identity
        
        if xFlag
            trFinv = trace(Finv);
        else
            trFinv = real(trace(B*Finv*B'));
        end
        
        fs = trFinv;
        
        if fs < fz            
            fz = fs;
            in = l;
        end
    end
    
    sel = [sel in];
    disp([num2str(length(sel)) ' sensors selected out of ' num2str(k) '. Objective: ' num2str(round(fz,3))])
    
end

iSel = sel;

end


