function A = real_diag(A)
% Replace the diagonal of a square matrix by its real part

n = size(A,1);
A(find(eye(n))) = real(diag(A));

end
