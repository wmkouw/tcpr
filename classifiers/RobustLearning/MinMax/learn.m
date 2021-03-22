% This function performs regular learning and evaluates the model
% Input: Xtr - (t x n) training data matrix, each row is an example
%        Xte - (te x n) test data matrix
%        Xref - (k x n) reference data matrix
%        ytr - (t x 1) training labels, +-1 if classification
%        yte - (te x 1) test labels
%        options - options for experiment
%          -- beta - (scalar) parameter for regularization on model theta
%          -- sigma - (scalar) kernel width for gaussian adversary
%          -- gamma - (scalar) parameter for soft moment matching constraints
%          -- type - (character) 'R' for regression, 'C' for classification
%          -- kernel - kernel type for model theta
%          -- learner_sigma - (scalar) kernel parameter for model theta
%          -- B - (scalar) bound on alpha
%          -- maxIter - (scalar) maximum # of iterations for solvers
%          -- w - (t x 1) specific weights for learning
% Output: theta - regular model
%         alpha - alphas chosen by the adversary
%           -- tr - alphas chosen for training points
%           -- te - alphas chosen for test points
%         lossTr - training loss of the model
%         lossTe - test loss of the model
%         lossTrAd - adversarial training loss of the model
%         lossTeAd - adversarial test loss of the model
%         errTr - training error of the model
%         errTe - test error of the model
%         errTrAd - adversarial training error of the model
%         errTeAd - adversarial test error of the model
% Last modified: July 28, 2014
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function [theta, alpha, lossTr, lossTe, lossTrAd, lossTeAd, errTr, errTe, errTrAd, errTeAd] =...
    learn(Xtr, Xte, Xref, ytr, yte, options)
[t, n] = size(Xtr);
beta = options.beta;
if(~isfield(options,'w'))
    options.w = ones(t,1)./t;
end

switch options.type
    case 'R'
        Func = @funcNonRobReg;
        lenTheta = t;
    case 'C'
        Func = @funcNonRobCls;
        lenTheta = t+1;
end

% kernels for learning
Ktr = options.kernel(Xtr, Xtr, options.learner_sigma);
Kte = options.kernel(Xte, Xtr, options.learner_sigma);
objFunc = @(theta)Func(theta, Ktr, ytr, beta, options.w);

theta0 = rand(lenTheta, 1);

% Set minFunc
if isempty(which('minFunc')); addpath(genpath('~/Dropbox/Libs/minFunc')); end
options.Display = 'valid';
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';

theta = minFunc(objFunc, theta0, options);
% ub = inf(lenTheta, 1);
% lb = -inf(lenTheta, 1);
% param = setParam(options);
%     [theta, obj, iter, numCall, flag] = pbm(objFunc, theta0, lb, ub, param);
%     [theta, obj, iter, numCall, flag] = lbfgsb(theta0, -inf(n,1), inf(n,1), objFunc, [], @genericcallback, param);
[lossTr, errTr] = evalu(Ktr, ytr, theta, options.type);
[lossTe, errTe] = evalu(Kte, yte, theta, options.type);

switch options.type
    case 'R'
        Func = @funcRobReg;
    case 'C'
        Func = @funcRobCls;
end
% (Gaussian) kernels for reweighing
Ktr_tr = gausskernel(Xtr, Xtr, options.sigma);
Ktr_ref = gausskernel(Xtr, Xref, options.sigma);
%     Ktr_ref = options.I;
Kref_ref = gausskernel(Xref, Xref, options.sigma);
[~, ~, alpha.tr] = Func(theta, Ktr, ytr, Ktr, Ktr_tr, Ktr_ref, Kref_ref, beta, options.gamma, options.B);
wTr = Ktr_ref*alpha.tr;
[lossTrAd, errTrAd] = evalu(Ktr, ytr, theta, options.type, wTr);
%     % First option: a different alpha
Kte_te = gausskernel(Xte, Xte, options.sigma);
Kte_ref = gausskernel(Xte, Xref, options.sigma);
[~, ~, alpha.te] = Func(theta, Kte, yte, Ktr, Kte_te, Kte_ref, Kref_ref, beta, options.gamma, options.B);
wTe = Kte_ref*alpha.te;
% Second option: the same alpha
%     alphaTe = alpha;
%     wTe = gausskernel(Xte,Xref,options.sigma)*alphaTe;
[lossTeAd, errTeAd] = evalu(Kte, yte, theta, options.type, wTe);
end
