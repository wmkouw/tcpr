function [iw,theta,gamma] = iwe_kmm(X,Z,varargin)
% Use Kernel Mean Matching to estimate importance weights.
%
% Based on: Huang, Smola, Gretton, Borgwardt & Schoelkopf (2007).
% Correcting Sample Selection Bias by unlabeled data.
%
% Wouter Kouw
% Last update: 2017-01-31

% Shapes
[NX,~] = size(X);
[NZ,~] = size(Z);

% Parse optionals
p = inputParser;
addOptional(p, 'mD', 'se');
addOptional(p, 'gamma', 0);
addOptional(p, 'theta', 1);
addOptional(p, 'eps', [0 0]);
addOptional(p, 'c', 1);
addOptional(p, 'lb', 0);
addOptional(p, 'ub', NX);
addOptional(p, 'nF', 2);
addOptional(p, 'nR', 1);
addOptional(p, 'hyp', '');
addOptional(p, 'nB', 5);
addOptional(p, 'lBo', round(NX./2));
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'yTol', 1e-5);
addOptional(p, 'cTol', 1e-5);
addOptional(p, 'maxIter', 2e3);
addOptional(p, 'solver', 'quadprog');
parse(p, varargin{:});

% Epsilon as a function of the upper bound B and the sample size
if isempty(p.Results.eps)
    eps = [p.Results.ub./sqrt(NX) p.Results.ub./sqrt(NX)];
else
    eps = p.Results.eps;
end

% Constraints
A = [ones(1,NX)./NX; -ones(1,NX)./NX];
b = [eps(1)+p.Results.c eps(2)-p.Results.c];
lb = p.Results.lb+zeros(NX,1);
ub = p.Results.ub+zeros(NX,1);

% Hyperparameter optimization
switch p.Results.hyp
    case 'mmd'
        [theta,gamma] = exp_kmm_hyp_mmd(X,Z, 'c', p.Results.c, 'nF', p.Results.nF, 'nR', p.Results.nR, 'eps', eps);
    case 'bootstrap'
        [theta,gamma] = exp_kmm_hyp_bootstrap(X,Z,'nB', p.Results.nB, 'lBo', p.Results.lBo, 'nR', 1);
    otherwise
        theta = p.Results.theta;
        gamma = p.Results.gamma;
end
% Report final hyperparameters
sprintf('sigma = %g',theta)
sprintf('gamma = %g',gamma)

% Compute kernels with optimized theta
switch p.Results.mD
    case 'ha'
        KXX = exp(-pdist2(X,X, 'hamming')./(2*theta.^2));
        kXZ = NX./NZ.*sum(exp(-pdist2(X,Z,'hamming')./(2*theta.^2)),2);
    case 'cb'
        KXX = exp(-pdist2(X,X, 'cityblock')./(2*theta.^2));
        kXZ = NX./NZ.*sum(exp(-pdist2(X,Z,'cityblock')./(2*theta.^2)),2);
    case 'se'
        KXX = exp(-pdist2(X,X, 'squaredeuclidean')./(2*theta.^2));
        kXZ = NX./NZ.*sum(exp(-pdist2(X,Z,'squaredeuclidean')./(2*theta.^2)),2);
    case 'ex'
        KXX = exp(X*X'./theta);
        kXZ = NX./NZ.*sum(exp(X*Z'./theta),2);
    case 'qd'
        KXX = (1+X*X'./2).^2;
        kXZ = NX./NZ.*sum((1+X*Z'./2).^2,2);
    otherwise
        error('Kernel type not recognized');
end

% Add weight regularization l2*||beta||^2
KXX = KXX + p.Results.gamma.*eye(NX);

% J(b) = 1/2*b'*KXX*b - k'*b + l2*||b||^2    s.t. b in [0,B], |mean(b)-1|<e
switch p.Results.solver
    case 'quadprog'
        % Optimization options
        options = optimoptions('quadprog', 'Display', 'final', ...
            'OptimalityTolerance', p.Results.yTol, ...
            'StepTolerance', p.Results.xTol, ...
            'ConstraintTolerance', p.Results.cTol, ...
            'maxIterations', p.Results.maxIter);
        
        % Run solver
        iw = quadprog(KXX,kXZ,A,b,[],[],lb, ub, [], options);
    case 'gurobi'
        % Objective
        model.Q = sparse(KXX);
        model.obj = -kXZ;
        
        % Inequality constraints
        model.A = sparse(A);
        model.rhs = b;
        model.sense = '<=';
        
        % Upper and lower bounds
        model.lb = p.Results.lb*ones(NX,1);
        model.ub = p.Results.ub*ones(NX,1);
        
       % Run solver
        result = gurobi(model);
        if strcmpi(result.status, 'optimal')
            iw = result.x;
        else
            error('Problem is infeasible');
        end
    case 'cvx'
        error('Not implemented yet');
    case 'tfocs'
        error('Not implemented yet');
    otherwise
        error('Solver not recognized');
end

end
