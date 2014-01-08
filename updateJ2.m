function J = updateJ2(rho, Z, Lam2, lambda1, J, tau, Lam3)
%using trace norm proximal operator to update J
[m, n] = size(J);
e = ones(m, n);
g = 2*e*(e*J - e + Lam3/rho);
T = Z + Lam2/rho + J - tau*g/rho;
[U, S, V] = svd(T);
S = diag(S);
S = max(S - lambda1*tau/(rho*(tau+1)), 0);
J = U * diag(S) * V';

end

