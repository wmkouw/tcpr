% Script to run domain adaptation experiments on heart disease data

% Include dependencies
addpath(genpath('../minFunc'));
addpath(genpath('../util'));
addpath(genpath('../../da-tools/'));

% Experimental parameters
prep = 'maxdiv';
nR = 1;

% Hyperparameters
lambda = 0.0;
alpha = 2;

% Optimization parameters
xTol = 1e-8;
maxIter = 2e3;

% Loop over all included classifiers
clfs = {'tca', 'kmm-lsq', 'rcsa', 'rba', 'tcp-lda', 'tcp-qda'};
for c = 1:length(clfs)

    exp_da_wAUS('prep', {prep}, 'nR', nR, 'clf', clfs{c}, ...
        'xTol', xTol, 'maxIter', maxIter, 'lambda', lambda, 'alpha', alpha);
end
