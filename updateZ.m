function Z = updateZ(rho, X, E, J, Z, Lam1, Lam2, lambda2, tau)
% using linearized ADMM to update Z
g = -X'*(X - X*Z - E + Lam1/rho);
Q = (Z - tau*g + tau*J - tau*Lam2/rho)/(1+tau);
thres = lambda2*tau/(rho*(1+tau));
Z = sign(Q).* max(abs(Q) - thres,0);

end

