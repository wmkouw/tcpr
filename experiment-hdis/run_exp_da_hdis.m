% Script to run domain adaptation experiments on heart disease data

% Include dependencies
addpath(genpath('../util'));

% Experimental parameters
prep = {'minusmin','maxdiv'};
nR = 1;

% Hyperparameters
lambda = 1e-3;
alpha = 2;
mu = 1./2;
nC = 8;

% Optimization parameters
xTol = 1e-8;
maxIter = 1e3;

% Save location
mkdir results
saveName = 'results/';

% Loop over all included classifiers
clfs = {'tca', 'kmm-lsq', 'rcsa', 'rba', 'tcp-ls', 'tcp-lda', 'tcp-qda'};
for c = 1:length(clfs)

    exp_da_hdis('prep', prep, 'nR', nR, 'clf', clfs{c}, ...
        'xTol', xTol, 'maxIter', maxIter, ...
        'lambda', lambda, 'alpha', alpha, 'mu', mu, 'nC', nC, ...
        'dataName', 'heart_disease', 'saveName', saveName);
end
