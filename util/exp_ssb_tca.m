function exp_ssb_tca(D,y,varargin)
% Sample selection bias experiment for Transfer Component Analysis

% Size
[M,~] = size(D);

% Parse arguments
p = inputParser;
addOptional(p, 'nN', min(10, round(M./2)));
addOptional(p, 'nM', M);
addOptional(p, 'setDiff', false);
addOptional(p, 'nC', 5);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'mu', 1/2);
addOptional(p, 'lambda', 1);
addOptional(p, 'sigma', 1);
addOptional(p, 'ssb', 'sdw');
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'saveName', []);
addOptional(p, 'viz', false);
parse(p, varargin{:});

% Number of sample sizes
lNN = length(p.Results.nN);
if isempty(p.Results.nM)
    lNM = 1;
else
    lNM = length(p.Results.nM);
end

% Check for column vector y
if ~iscolumn(y); y = y'; end

% Labels
labels = unique(y);
K = numel(labels);

% Preallocate
theta = cell(p.Results.nR,lNN,lNM);
e = NaN(p.Results.nR,lNN,lNM);
R = NaN(p.Results.nR,lNN,lNM);
AUC = NaN(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
P = cell(p.Results.nR,lNN,lNM);

for r = 1:p.Results.nR
    disp(['Running repeat ' num2str(r) '/' num2str(p.Results.nR)]);
    
    for n = 1:lNN
        
        % Select samples
        switch p.Results.ssb
            case 'nn'
                ix = ssb_nn(D,y,p.Results.nN(n), 'viz', p.Results.viz);
            case 'sdw'
                ix = ssb_sdw(D,y,p.Results.nN(n), 'viz', p.Results.viz);
            otherwise
                error(['Selection bias type ' p.Results.ssb ' unknown']);
        end
        
        % Select source
        X = D(ix,:);
        yX = y(ix);
        
        % Select target
        if p.Results.setDiff
            Z = D(setdiff(1:M,ix),:);
            yZ = y(setdiff(1:M,ix));
        else
            Z = D;
            yZ = y;
        end
        
        for m = 1:lNM
            
            % Select target samples
            if ~isempty(p.Results.nM)
                ixNM = randsample(1:M, p.Results.nM(m), false);
            else
                ixNM = 1:size(Z,1);
            end
            
            % Cross-validate regularization parameter
            if isempty(p.Results.lambda)
                disp(['Cross-validating for regularization parameter']);
                
                % Set range of regularization parameter
                Lambda = [0 10.^[-6:1:3]];
                R_la = zeros(1,length(Lambda));
                for la = 1:length(Lambda)
                    
                    % Split folds
                    ixf = randsample(1:p.Results.nF, p.Results.nN(n), true);
                    for f = 1:p.Results.nF
                        
                        % Train on included folds
                        theta_f = tca(X(ixf~=f,:),yX(ixf~=f),Z(ixNM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'nC', p.Results.nC, 'l2', Lambda(la));
                        
                        % Evaluate on held-out source folds (error)
                        lf = sum(ixf==f);
                        [Pf,KXZ] = tc(X(ixf==f,:)', Z(ixNM,:)', 'sigma', p.Results.sigma, 'mu', p.Results.mu, 'nC', p.Results.nC);
                        [~,pred_f] = max([KXZ(1:lf,:)*Pf ones(lf,1)]*theta_f, [], 2);
                        R_la(la) = mean(labels(pred_f) ~= yX(ixf==f),1);
                    end
                end
                % Select minimal
                R_la(isinf(R_la)) = NaN;
                [~,ixMinLambda] = min(R_la);
                lambda = Lambda(ixMinLambda);
            else
                lambda = p.Results.lambda;
            end
            disp(['\lambda = ' num2str(lambda)]);
            
            % Call classifier and evaluate
            [theta{r,n,m},P{r,n,m},R(r,n,m),e(r,n,m),pred{r,n,m},post{r,n,m},AUC(r,n,m)] = tca(X, yX, Z(ixNM,:),'yZ', yZ(ixNM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'nC', p.Results.nC, 'sigma', p.Results.sigma, 'l2', lambda);
        end
    end
end

% Write results
di = 1; while exist([p.Results.saveName 'results_ssb_tca_' num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = [p.Results.saveName 'results_ssb_tca_' num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R','e', 'pred', 'post', 'AUC','p', 'P');

end
