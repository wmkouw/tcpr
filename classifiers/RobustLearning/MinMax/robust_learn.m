function [theta, alpha, weight, obj, lossTrAd, lossTeAd, lossTe, errTrAd, errTeAd, errTe] =...
    robust_learn(Xtr, Xte, Xref, ytr, yte, options)
% This function performs robust learning (or RCSA) and evaluates the model
% Input: Xtr - (t x n) training data matrix, each row is an example
%        Xte - (te x n) test data matrix
%        Xref - (k x n) reference data matrix
%        ytr - (t x 1) training labels, +-1 if classification
%        yte - (te x 1) test labels
%        options - options for experiment
%          -- beta - (scalar) parameter for regularization on model theta
%          -- sigma - (scalar) kernel width for gaussian adversary
%          -- gamma - (scalar) parameter for soft moment matching constraints (true for RCSA)
%          -- type - (character) 'R' for regression, 'C' for classification
%          -- kernel - kernel type for model theta
%          -- learner_sigma - (scalar) kernel parameter for model theta
%          -- B - (scalar) bound on alpha
%          -- maxIter - (scalar) maximum # of iterations for solvers
%          -- w - (t x 1) specific weights for learning
% Output: theta - robust model (or RCSA model)
%         alpha - alphas chosen by the adversary
%           -- tr - alphas chosen for training points
%           -- te - alphas chosen for test points
%         weight - weights for points
%           -- robTr - weights for training points
%           -- robTe - weights for test points
%         obj - robust learning final objective value
%         lossTrAd - adversarial training loss of the model
%         lossTeAd - adversarial test loss of the model
%         lossTe - test loss of the model
%         errTrAd - adversarial training error of the model
%         errTeAd - adversarial test error of the model
%         errTe - test error of the model
% Last modified: July 28, 2014
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta

[t, n] = size(Xtr);
beta = options.beta;
gamma = options.gamma;
B = options.B;

switch options.type
    case 'R'
        Func = @funcRobReg;
        lenTheta = t;
    case 'C'
        Func = @funcRobCls;
        lenTheta = t+1;
end

Ktr_tr = gausskernel(Xtr, Xtr, options.sigma);
Ktr_ref = gausskernel(Xtr, Xref, options.sigma);
%     Ktr_ref = options.I;
Kref_ref = gausskernel(Xref, Xref, options.sigma);
Ktr = options.kernel(Xtr, Xtr, options.learner_sigma);
Kte = options.kernel(Xte, Xtr, options.learner_sigma);
objFunc = @(theta)Func(theta, Ktr, ytr, Ktr, Ktr_tr, Ktr_ref, Kref_ref, beta, gamma, B);

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
% [theta, obj, iter, numCall, flag] = pbm(objFunc, theta0, lb, ub, param);
%     [theta, obj, iter, numCall, flag] = lbfgsb(theta0, -inf(n,1), inf(n,1), objFunc, [], @genericcallback, param);
[~, ~, alpha.tr] = Func(theta, Ktr, ytr, Ktr, Ktr_tr, Ktr_ref, Kref_ref, beta, gamma, B);
weight.robTr = Ktr_ref*alpha.tr;

if nargout>3
    [lossTrAd, errTrAd] = evalu(Ktr, ytr, theta, options.type, weight.robTr);
    %     % First option: a different alpha
    Kte_te = gausskernel(Xte, Xte, options.sigma);
    Kte_ref = gausskernel(Xte, Xref, options.sigma);
    [~, ~, alpha.te] = Func(theta, Kte, yte, Ktr, Kte_te, Kte_ref, Kref_ref, beta, gamma, B);
    weight.robTe = Kte_ref*alpha.te;
    % Second option: the same alpha
    %     alphaTe = alpha;
    %     wTe = gausskernel(Xte,Xref,options.sigma)*alphaTe;
    [lossTeAd, errTeAd] = evalu(Kte, yte, theta, options.type, weight.robTe);
    [lossTe, errTe] = evalu(Kte, yte, theta, options.type);
end
end
