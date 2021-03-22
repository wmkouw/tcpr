% Toy function to set parameters for solver
function param = setParam(options)
    param.maxIter = options.maxIter;     % max number of iterations
    param.maxLsIter = 20;   % max number of line search steps in each iteration
    param.maxBdl = 10;      % max number of bundles to keep
    param.maxFnCall = options.maxIter;  % max number of calling the function
    param.tolCon = 1e-5;      % tolerance of constraint satisfaction
    param.tolFun = options.tol;   % final objective function accuracy parameter
    param.relCha = options.tol;      % tolerance of constraint satisfaction
    param.tolPG = options.tol;   % final objective function accuracy parameter
    param.m = 50;
end
