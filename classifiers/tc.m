function [M,K] = tc(X,Z,varargin)
% Transfer components

% Parse input
p = inputParser;
addParameter(p, 'sigma', 1);
addParameter(p, 'mu', 1);
addParameter(p, 'nC', 1);
parse(p, varargin{:});

% Shapes
[~,n] = size(X);
[~,m] = size(Z);

% Form block kernel
K = K_block(X',Z', 'sigma', p.Results.sigma);

% Objective function
[M,~] = eigs((eye(n+m)+p.Results.mu*K*[ones(n)./n.^2 -ones(n,m)./(n*m); ...
    -ones(m,n)./(n*m) ones(m)./m.^2]*K)\(K*((1-1./(n+m)).*eye(n+m))*K), p.Results.nC);

% Ensure outcome is real
M = real(M);

end
