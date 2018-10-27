% Script to run domain adaptation experiments on heart disease data

% Include dependencies
addpath(genpath('../data'));
addpath(genpath('../util'));

% Experimental parameters
prep = {'minusmin','maxdiv', 'impute0'};
nR = 1;

% Hyperparameters
lambda = 1.0;
alpha = .0001;
mu = 1./2;
nC = 8;

% Optimization parameters
xTol = 1e-20;
maxIter = 5e4;
lr = 'const';

% Save location
mkdir results
saveName = 'results/';

% Loop over all included classifiers
clfs = {'slda', 'sqda', 'tca', 'kmm-lsq', 'rcsa', 'rba', 'tcp-lda', 'tcp-qda'};
for c = 1:length(clfs)

    exp_da_hdis('prep', prep, 'nR', nR, 'clf', clfs{c}, ...
        'xTol', xTol, 'maxIter', maxIter, 'lr', lr, ...
        'lambda', lambda, 'alpha', alpha, 'mu', mu, 'nC', nC, ...
        'dataName', 'heart_disease', 'saveName', saveName);
end
