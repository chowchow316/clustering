function V = CalLap(Z, k)
[m, n] = size(Z);
A = abs(Z') + abs(Z);
D = diag(sum(A, 2));
D2 = eye(n)/sqrt(D);
L = eye(n) - D2*A*D2;
[V, D]  = eig(L);
V = V(:, 1:k);
end