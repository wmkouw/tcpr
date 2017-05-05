function run_exp_da_hdis(varargin)
% Run domain adaptation experiments on the heart disease dataset

addpath(genpath('../minFunc'));
addpath(genpath('../util'));

% Parse hyperparameters
p = inputParser;
addOptional(p, 'cix', []);
addOptional(p, 'clf', 'slda');
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'lr', 'geom');
addOptional(p, 'prep_a', {''});
addOptional(p, 'prep_d', {''});
addOptional(p, 'maxIter', 1e4);
addOptional(p, 'xTol', 1e-10);
addOptional(p, 'alpha', 1);
addOptional(p, 'lambda', []);
addOptional(p, 'gamma', .1);
addOptional(p, 'useGamma', true);
addOptional(p, 'iwe', 'kmm');
addOptional(p, 'clip', realmax);
addOptional(p, 'NN', []);
addOptional(p, 'NM', []);
addOptional(p, 'datnm', 'hdis_imp0');
parse(p, varargin{:});

% Report which classifier
disp(['Running: ' p.Results.clf]);

% Load dataset
load(p.Results.datnm)
disp(['Loaded dataset: ' p.Results.datnm]);

% Preprocess data
D = da_prep(D, p.Results.prep_a);

% Source-Target combinations
nD = length(domains)-1;
cc = [nchoosek(1:nD,2); fliplr(nchoosek(1:nD,2))];
if isempty(p.Results.cix)
    cmbl = 1:size(cc,1);
else
    cmbl = p.Results.cix;
end

for n = cmbl
    
    % Split out source and target
    ixS = domains(cc(n,1))+1:domains(cc(n,1)+1);
    ixT = domains(cc(n,2))+1:domains(cc(n,2)+1);
    
    % Run estimator
    switch p.Results.clf
        % Source classifiers
        case 'slda'
            exp_da_sda(D(ixS, :),y(ixS), D(ixT,:), y(ixT), 'NN', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda, 'clf', 'lda');
        case 'sqda'
            exp_da_sda(D(ixS, :),y(ixS), D(ixT,:), y(ixT), 'NN', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda, 'clf', 'qda');
        case 'sls'
            exp_da_sls(D(ixS, :),y(ixS), D(ixT,:), y(ixT), 'NN', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda);
        case 'lr'
            exp_da_lr(D(ixS,:), y(ixS), D(ixT,:), y(ixT),'N', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'lambda',lambda);
        case 'svm'
            exp_da_svm(D(ixS,:), y(ixS), D(ixT,:), y(ixT),'N', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'lambda',lambda);
            
            % Domain adaptive classifiers
        case 'tce-lda'
            exp_da_mcpl(D(ixS,:), y(ixS), D(ixT,:), y(ixT), 'NN', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'alpha', p.Results.alpha, 'lambda', p.Results.lambda, 'clf', 'tce-lda');
        case 'tce-qda'
            exp_da_mcpl(D(ixS,:), y(ixS), D(ixT, :), y(ixT), 'NN', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'alpha', p.Results.alpha, 'lambda', p.Results.lambda, 'clf', 'tce-qda', 'lr', p.Results.lr);
        case 'tce-ls'
            exp_da_mcpl(D(ixS,:), y(ixS), D(ixT, :), y(ixT), 'NN', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'alpha', p.Results.alpha, 'lambda', p.Results.lambda, 'clf', 'tce-ls');
        case 'kmm-lsq'
            exp_da_iwc(D(ixS,:), y(ixS), D(ixT, :), y(ixT), 'NN', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'iwe', 'kmm', 'clf', 'lsq');
        case 'kmm-lr'
            exp_da_iwc(D(ixS,:), y(ixS), D(ixT, :), y(ixT), 'NN', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'iwe', 'kmm', 'clf', 'lr');
        case 'rba'
            exp_da_rba(D(ixS,:), y(ixS), D(ixT,:), y(ixT), 'NN', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'gamma', p.Results.gamma, 'lambda', p.Results.lambda,'clip', p.Results.clip, 'iwe', p.Results.iwe);        
        case 'rcsa'
            exp_da_rcsa(D(ixS,:), y(ixS), D(ixT, :), y(ixT), 'NN', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'useGamma', p.Results.useGamma, 'lambda', p.Results.lambda);
        case 'rda'
            exp_da_rda(D(ixS,:), y(ixS), D(ixT,:), y(ixT), 'N', p.Results.NN, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'lambda', p.Results.lambda,'clip', p.Results.clip);
            
            % Target classifiers
        case 'tlda'
            exp_da_tda(D(ixT,:), y(ixT), 'NM', p.Results.NM, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda, 'clf', 'lda');
        case 'tqda'
            exp_da_tda(D(ixT,:), y(ixT), 'NM', p.Results.NM, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda, 'clf', 'qda');
        case 'tls'
            exp_da_tls(D(ixT, :),y(ixT),'NM', p.Results.NM, 'nR', p.Results.nR, 'nF', p.Results.nF, 'prep', p.Results.prep_d, 'svnm', [p.Results.datnm '_prepa_' p.Results.prep_a{logical(cellfun(@isstr, p.Results.prep_a))} '_prepd_' p.Results.prep_d{logical(cellfun(@isstr, p.Results.prep_d))} '_cc' sprintf('%1i', n) '_repeat' num2str(p.Results.nR) '_'], 'lambda',p.Results.lambda);
        otherwise
            disp(['Classifier not recognized']);
    end
end

% exit;

end
