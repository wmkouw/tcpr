function run_exp_ssb_mamm(varargin)
% Run sample selection bias experiments on the mammographic masses dataset

addpath(genpath('../minFunc'));
addpath(genpath('../util'));

% Parse hyperparameters
p = inputParser;
addOptional(p, 'clf', 'slda');
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'lr', 'geom');
addOptional(p, 'prep', {''});
addOptional(p, 'maxIter', 1e4);
addOptional(p, 'xTol', 1e-8);
addOptional(p, 'alpha', 1);
addOptional(p, 'lambda', 0);
addOptional(p, 'gamma', .1);
addOptional(p, 'useGamma', true);
addOptional(p, 'iwe', 'kliep');
addOptional(p, 'clip', realmax);
addOptional(p, 'nN', 10);
addOptional(p, 'nM', []);
addOptional(p, 'setDiff', false);
addOptional(p, 'datnm', 'mamm_imp0');
parse(p, varargin{:});

% Report which classifier
disp(['Running: ' p.Results.clf]);

% Load dataset
load(p.Results.datnm)
disp(['Loaded dataset: ' p.Results.datnm]);

% Preprocess data
D = da_prep(D, p.Results.prep);

% Run estimator
switch p.Results.clf
    % Source classifiers
    case 'slda'
        exp_ssb_sda(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda, 'clf', 'lda');
    case 'sqda'
        exp_ssb_sda(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda, 'clf', 'qda');
    case 'sls'
        exp_ssb_sls(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda);
    case 'lr'
        exp_ssb_lr(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'lambda',lambda);
    case 'svm'
        exp_ssb_svm(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'lambda',lambda);
        
        % Adaptive classifiers
    case 'tcp-lda'
        exp_ssb_tcp(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'alpha', p.Results.alpha, 'lambda', p.Results.lambda, 'clf', 'tcp-lda');
    case 'tcp-qda'
        exp_ssb_tcp(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'alpha', p.Results.alpha, 'lambda', p.Results.lambda, 'clf', 'tcp-qda', 'lr', p.Results.lr);
    case 'tcp-ls' 
        exp_ssb_tcp(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'alpha', p.Results.alpha, 'lambda', p.Results.lambda, 'clf', 'tcp-ls');
    case 'kmm-lsq'
        exp_ssb_iwc(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'iwe', 'kmm', 'clf', 'lsq');
    case 'kmm-lr'
        exp_ssb_iwc(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'iwe', 'kmm', 'clf', 'lr');
    case 'rba'
        exp_ssb_rba(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'gamma', p.Results.gamma, 'lambda', p.Results.lambda,'clip', p.Results.clip, 'iwe', p.Results.iwe);
    case 'rcsa'
        exp_ssb_rcsa(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'useGamma', p.Results.useGamma, 'lambda', p.Results.lambda);
    case 'rda'
        exp_ssb_rda(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'lambda', p.Results.lambda,'clip', p.Results.clip);
        
        % Target classifiers
    case 'tlda'
        exp_ssb_tda(D,y, 'nM', p.Results.nN, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda, 'clf', 'lda');
    case 'tqda'
        exp_ssb_tda(D,y, 'nM', p.Results.nN, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda, 'clf', 'qda');
    case 'tls'
        exp_ssb_tls(D,y, 'nM', p.Results.nN, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'svnm', ['ssb_' p.Results.datnm '_prep_' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_rep' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda);
    otherwise
        disp(['Classifier not recognized']);
end


end
