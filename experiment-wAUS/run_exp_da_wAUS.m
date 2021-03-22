% Script to run domain adaptation experiments on heart disease data

% Include dependencies
addpath(genpath('data'));
addpath(genpath('..\classifiers'));
addpath(genpath('..\opt'));
addpath(genpath('..\util'));

% Experimental parameters
prep = {'minusmin', 'maxdiv'};
nR = 1;

% Hyperparameters
lambda = .1;
alpha = .01;
gamma = .001;

% Optimization parameters
xTol = 1e-20;
maxIter = 5e3;
lr = 'lin';

% Loop over all included classifiers
clfs = {'slda', 'sqda', 'tca', 'kmm-lsq', 'rcsa', 'rba', 'tcp-lda', 'tcp-qda'};
for c = 1:length(clfs)

    exp_da_wAUS('prep', prep, 'nR', nR, 'clf', clfs{c}, ...
                'xTol', xTol, 'maxIter', maxIter, 'lr', lr, ...
                'lambda', lambda, 'alpha', alpha, 'gamma', gamma);
end
