% Script to run domain adaptation experiments on mammographic masses data

% Include dependencies
addpath(genpath('../util'));

% Experimental parameters
prep = {'minusmin','maxdiv'};
nR = 1;
nN = 50;
nM = [];

% Hyperparameters
lambda = 1e-3;
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

    exp_ssb_mamm('prep', prep, 'nR', nR, 'nN', nN, 'nM', nM, 'clf', clfs{c}, ...
        'xTol', xTol, 'maxIter', maxIter, 'gamma', gamma, 'lambda', lambda, ...
        'alpha', alpha, 'mu', mu, 'sigma', sigma, 'nC', nC, ...
        'saveName', saveName);
end

