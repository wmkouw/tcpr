function [iw] = iw_KMM(X,Z,varargin)
% Use Kernel Mean Matching to estimate weights for importance weighting.
%
% Jiayuan Huang, Alex Smola, Arthur Gretton, Karsten Borgwardt & Bernhard
% Schoelkopf. Correcting Sample Selection Bias by unlabeled data.

% Parse optionals
p = inputParser;
addOptional(p, 'theta', 0);
addOptional(p, 'kernel', 'mvn');
addOptional(p, 'clip', Inf);
addOptional(p, 'B', 2);
parse(p, varargin{:});

% Shapes
[n,d] = size(X);
[m,~] = size(Z);

switch p.Results.kernel
    case 'mvn'
        % Multivariate Normal
        
        % Bandwidth matrix
        if p.Results.theta==0
            % Silverman's rule
            si = std(X,[],1);
            H = spdiags(((4./(d+2))^(1/d+4)*n^(-1/(d+4))*si)^2, 0, d, d);
        elseif p.Results.theta==-1
            % Scott's rule
            si = std(X,[],1);
            H = spdiags((n^(-1/(d+4))*si)^2,0,d,d);
        elseif isvector(p.Results.theta)
            H = spdiags(p.Results.theta,0,d,d);
        elseif isscalar(p.Results.theta)
            H = p.Results.theta*speye(d);
        else
            H = p.Results.theta;
        end

        % Compute kernels
        K = det(H)^(-1/2).*(2*pi)^(-d./2)*exp(-X/H*X'./2);
        k = n./m*sum(det(H)^(-1/2).*(2*pi)^(-d./2)*exp(-X/H*Z'./2),2);
        
    case 'rbf'
        % Radial basis function
        
        % Silverman's rule
        if p.Results.theta==0;
            % Silverman's rule
            si = std(X,[],1);
            theta = spdiags(((4./(d+2))^(1/d+4)*n^(-1/(d+4))*si)^2, 0, d, d);
            
            % Compute kernels
            K = exp(-pdist2(X, X, 'Mahalanobis', theta));
            k = n./m*sum(exp(-pdist2(X, Z, 'Mahalanobis', theta)),2);
            
        else
            theta = p.Results.theta;
            
            % Compute kernels
            K = exp(-pdist2(X, X)./theta);
            k = n./m*sum(exp(-pdist2(X, Z)./theta),2);
        end
        
    case 'diste'
        % Calculate Euclidean distances
        K = pdist2(X, X);
        k = n./m*sum(pdist2(X, Z),2);
end

% Solve quadratic program
options.Display = 'final';
% options.Algorithm = 'active-set';
options.TolCon = 1e-5;
eps = p.Results.B./sqrt(n);
A = [ones(1,n); -ones(1,n)];
b = [n*(eps+1); n*(eps-1)];
lb = zeros(n,1);
ub = p.Results.B*ones(n,1);
iw = quadprog(K,k,A,b,[],[],lb, ub, [], options);

% Weight clipping
iw = min(p.Results.clip,max(0,iw));

end
