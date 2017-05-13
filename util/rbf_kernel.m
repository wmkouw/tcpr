function K = rbf_kernel(X,Z,varargin)
% Compute block radial-basis function kernel

p = inputParser;
addParameter(p, 'theta', 1);
parse(p, varargin{:});

% Source-target kernel
Kst = exp(-pdist2(X, Z)/(2*p.Results.theta.^2));

% Compute block-diagonal kernel
K = [exp(-pdist2(X, X)/(2*p.Results.theta.^2)) Kst; Kst' exp(-pdist2(Z, Z)/(2*p.Results.theta.^2))];


end
