function exp_ssb_iono(varargin)
% Sample selection bias experiments on the ionosphere dataset

% Parse hyperparameters
p = inputParser;
addOptional(p, 'clf', 'tcp-ls');
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'lr', 'geom');
addOptional(p, 'prep', {''});
addOptional(p, 'maxIter', 1e4);
addOptional(p, 'xTol', 1e-8);
addOptional(p, 'alpha', 2);
addOptional(p, 'sigma', 1);
addOptional(p, 'lambda', 0);
addOptional(p, 'gamma', .1);
addOptional(p, 'mu', 1);
addOptional(p, 'nC', 5);
addOptional(p, 'useGamma', true);
addOptional(p, 'iwe', 'kmm');
addOptional(p, 'clip', realmax);
addOptional(p, 'nN', 10);
addOptional(p, 'nM', []);
addOptional(p, 'setDiff', false);
addOptional(p, 'dataName', 'iono');
addOptional(p, 'saveName', '');
parse(p, varargin{:});

% Report which classifier
disp(['Running: ' p.Results.clf]);

% Load dataset
try
    load(p.Results.dataName)
catch
    cd('../data/iono')
    [D,y] = get_iono('save', true);
    copyfile iono.mat ../../experiment-iono/
    cd('../../experiment-iono')
end 
disp(['Loaded dataset: ' p.Results.dataName]);

% Preprocess data
D = da_prep(D, p.Results.prep);

% Select adaptive approach
switch p.Results.clf
    case 'tcp-lda'
        exp_ssb_tcp(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'saveName', [p.Results.saveName p.Results.dataName '_prep' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_nR' num2str(p.Results.nR) '_'], 'alpha', p.Results.alpha, 'lambda', p.Results.lambda, 'clf', 'tcp-lda', 'lr', p.Results.lr);
    case 'tcp-qda'
        exp_ssb_tcp(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'saveName', [p.Results.saveName p.Results.dataName '_prep' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_nR' num2str(p.Results.nR) '_'], 'alpha', p.Results.alpha, 'lambda', p.Results.lambda, 'clf', 'tcp-qda', 'lr', p.Results.lr);
    case 'tcp-ls' 
        exp_ssb_tcp(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'saveName', [p.Results.saveName p.Results.dataName '_prep' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_nR' num2str(p.Results.nR) '_'], 'alpha', p.Results.alpha, 'lambda', p.Results.lambda, 'clf', 'tcp-ls');
    case 'kmm-lsq'
        exp_ssb_iwc(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'saveName', [p.Results.saveName p.Results.dataName '_prep' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_nR' num2str(p.Results.nR) '_'], 'iwe', 'kmm', 'clf', 'lsq', 'gamma', p.Results.gamma);
    case 'kmm-lr'
        exp_ssb_iwc(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'saveName', [p.Results.saveName p.Results.dataName '_prep' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_nR' num2str(p.Results.nR) '_'], 'iwe', 'kmm', 'clf', 'lr', 'gamma', p.Results.gamma);
    case 'rba'
        exp_ssb_rba(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'saveName', [p.Results.saveName p.Results.dataName '_prep' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_nR' num2str(p.Results.nR) '_'], 'gamma', p.Results.gamma, 'lambda', p.Results.lambda,'clip', p.Results.clip);
    case 'rcsa'
        exp_ssb_rcsa(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'saveName', [p.Results.saveName p.Results.dataName '_prep' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_nR' num2str(p.Results.nR) '_'], 'useGamma', p.Results.useGamma, 'lambda', p.Results.lambda);
    case 'tca'
        exp_ssb_tca(D,y, 'nN', p.Results.nN, 'nM', p.Results.nM, 'setDiff', p.Results.setDiff, 'nR', p.Results.nR, 'nF', p.Results.nF, 'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'saveName', [p.Results.saveName p.Results.dataName '_prep' p.Results.prep{logical(cellfun(@isstr, p.Results.prep))} '_nR' num2str(p.Results.nR) '_'], 'mu', p.Results.mu, 'nC', p.Results.nC, 'lambda', p.Results.lambda, 'sigma', p.Results.sigma);
    otherwise
        disp(['Classifier not recognized']);
end

end

