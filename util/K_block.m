function K = K_block(X,Z,varargin)
% Compute block kernel

p = inputParser;
addParameter(p, 'sigma', 1);
addParameter(p, 'kernel', 'rbf');
parse(p, varargin{:});

switch p.Results.kernel
    case 'rbf'
        % Source-source kernel
        KXX = exp(-pdist2(X, X)/(2*p.Results.sigma.^2));

        % Source-target kernel
        KXZ = exp(-pdist2(X, Z)/(2*p.Results.sigma.^2));

        % Target-target kernel
        KZZ = exp(-pdist2(Z, Z)/(2*p.Results.sigma.^2));
    otherwise
        error('Kernel not recognized');
end        

% Form and return block
K = [KXX KXZ; KXZ' KZZ];

end
