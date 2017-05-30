% Script to run domain adaptation experiments on heart disease data

% Include dependencies
addpath(genpath('../util'));

% Experimental parameters
prep = {'minusmin','maxdiv', 'impute0'};
nR = 1;
nN = 50;
nM = [];

% Hyperparameters
lambda = [];
sigma = 1;
alpha = 2;
gamma = 1;
mu = 1./2;
nC = 8;

% Optimization parameters
xTol = 1e-8;
maxIter = 2e3;

% Save location
mkdir results
saveName = 'results/';

% Loop over all included classifiers
clfs = {'tca', 'kmm-lsq', 'rcsa', 'rba', 'tcp-ls', 'tcp-lda', 'tcp-qda'};
for c = 1:length(clfs)

    exp_ssb_pima('prep', prep, 'nR', nR, 'nN', nN, 'nM', nM, 'clf', clfs{c}, ...
        'xTol', xTol, 'maxIter', maxIter, 'gamma', gamma, 'lambda', lambda, ...
        'alpha', alpha, 'mu', mu, 'sigma', sigma, 'nC', nC, ...
        'dataName', 'pima', 'saveName', saveName);
end

