%% This code aims to solve the problem
%  min 1/2|X-XZ|^2 + lambda1|Z|_* + lambda2|Z|_1
%  Introduce auxiliary variables J and E, 
%  min 1/2|E|^2 + lambda1|J|_* + lambda2|Z|_1
%  s.t. X-XZ = E,
%       Z = J.
%  There are three subproblems: update E, J and Z.
%  By Qian Sun, 1/4/2013

clc;
clear all;
close all;

flag = 4; % 1- synthetic; 2-FaceB; 3-two circles

% each column is a sample
%load '../datasets/YaleFacesB5810.mat';
if flag == 1
    load syn.mat;
    X = X1';
	r = 3;
	%X = X';
    %[IDX, C, sumd, D] = kmeans(X1, 2);
end

if flag == 2
    load '../../datasets/YaleFacesB5810.mat';
	r = 3;
	labels = [ones(64, 1); 2*ones(64, 1); 3*ones(64,1)];
%     X = Y;
end
 
if flag == 3
    load '../../datasets/twocircles.mat';
    X = X';
	r = 2;
%     load EmbedCircle.mat;
%     X = U';
end

if flag == 4
	load '../../datasets/YaleB_48_42.mat';
	X = DATA;
	r = 38;
	labels = labels';
end

if flag == 5
	load '../../datasets/AR_55_40.mat';
	X = DATA;
	r = 100;
end

if flag == 6
	load '../../datasets/MPIE_50_41.mat';
	X = DATA;
	r = 286;	
end

if flag == 7
	load '../../datasets/covtype.mat';
	X = DATA;
	clear DATA;
	r = 7;	
end


[nrow, ncol] = size(X);

% initialize E, J, Z 
% and lagrangian multipliers Lam1, Lam2
Z = zeros(ncol, ncol);%rand(ncol, ncol);
J = Z;
E = X - X*Z;
Lam1 = zeros(nrow, ncol);
Lam2 = zeros(ncol, ncol);
Lam3 = zeros(ncol, ncol);

% set parameter
rho = 5;%2
t = 1.2; %adaptive t= 1.1~1.2;
MaxIter = 200;
tol = 1e-5;
lambda1 = 0.5;%.5;
lambda2 = 0.5; %0.5;
[~, sigma, ~] = svd(X, 'econ');
tau = 1/(1.02*max(diag(sigma))^2);
funVal = zeros(MaxIter, 1);
rho_max = 10e3;
e = ones(ncol, ncol);

% stop criterion
tic
for iter = 1: MaxIter
	
    disp(iter);
    Z_old = Z;
    % adaptive penalty
    rho = min(t*rho, rho_max);
    % update E
    E = (rho*X - rho*X*Z + Lam1)/(1+rho);
    % update J
    %J = updateJ(rho, Z, Lam2, lambda1);
    J = updateJ2(rho, Z, Lam2, lambda1, J, tau, Lam3);
	% update Z
    Z = updateZ(rho, X, E, J, Z, Lam1, Lam2, lambda2, tau);
    Z_new = Z;

    % calculate the function value
    funVal(iter) = 1/2*norm(E, 'fro')^2 + lambda1*sum(sum(abs(Z))) + trace(Lam1'*(X-X*Z-E)) ...
            + trace(Lam2'*(Z-J)) + rho/2*(norm(X - X*Z - E, 'fro') + norm(J - Z, 'fro'))...
			+trace(Lam3'*(e*J - e)) + rho/2*norm(e*J - e, 'fro');
	disp(funVal(iter));
        
    % check convergence
    if (iter >= 2 && norm(X - X*Z - E, 'fro') <= tol * max(norm(X, 'fro'), 1)...
                && norm(Z - J, 'fro') <= tol * max(norm(X, 'fro'), 1)) %&& norm(Z_new - Z_old, 'fro') <= tol  * max(norm(X, 'fro'), 1))
        funVal(iter + 1:end) = [];
        break;      
    end
    
    % update Lam1, Lam2
    Lam1 = Lam1 + rho*(X - X*Z - E);
    Lam2 = Lam2 + rho*(Z - J);
	Lam3 = Lam3 + rho*(e*J - e);

end
toc
% norm(X - X*Z - E, 'fro')
% norm(X - X*Z, 'fro')
% norm(Z - J, 'fro')

V = CalLap(Z, r);
Cind = kmeans(V, r); 

newLabel = bestMap(labels, Cind);

acc = nnz(newLabel==labels)/length(labels)

