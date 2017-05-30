function exp_ssb_iwc(D,y,varargin)
% Sample selection bias experiment for importance-weighted classifier

% Size
[M,~] = size(D);

% Parse arguments
p = inputParser;
addOptional(p, 'clf', 'lsq');
addOptional(p, 'nN', min(10, round(M./2)));
addOptional(p, 'nM', M);
addOptional(p, 'setDiff', false);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'gamma', 1);
addOptional(p, 'lambda', 1);
addOptional(p, 'clip', 100);
addOptional(p, 'ssb', 'sdw');
addOptional(p, 'iwe', 'kmm')
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'saveName', []);
addOptional(p, 'viz', false);
parse(p, varargin{:});

% Check for column vector y
if ~iscolumn(y); y = y'; end

% Labeling
labels = unique(y);
K = numel(labels);
if K>2; error('Binary classification only'); end

% Number of sample sizes
lNN = length(p.Results.nN);
if isempty(p.Results.nM)
    lNM = 1;
else
    lNM = length(p.Results.nM);
end

% Preallocate
theta = cell(p.Results.nR,lNN,lNM);
e = NaN(p.Results.nR,lNN,lNM);
R = NaN(p.Results.nR,lNN,lNM);
AUC = NaN(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
iw = cell(p.Results.nR,lNN,lNM);

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
                Lambda = [0 10.^[-6:2:0]];
                R_la = zeros(1,length(Lambda));
                for la = 1:length(Lambda)
                    
                    % Split folds
                    ixf = randsample(1:p.Results.nF, p.Results.nN(n), true);
                    for f = 1:p.Results.nF
                        
                        % Augment held-out fold
                        Xa = [X(ixf==f,:) ones(length(ixf==f),1)];
                        yXf = yX(ixf==f);
                        
                        switch p.Results.clf
                            case 'lsq'
                                % Train on included folds
                                theta_f = iwc(X(ixf~=f,:),yX(ixf~=f),Z(ixNM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'lambda', Lambda(la), 'clf', 'lsq', 'iwe', p.Results.iwe, 'gamma', p.Results.gamma);
                                
                                % Evaluate on held-out source folds (MSE)
                                R_la(la) = mean((Xa*theta_f - yXf).^2,1);
                            case 'lr'
                                % Train on included folds
                                theta_f = iwc(X(ixf~=f,:),yX(ixf~=f),Z(ixNM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'lambda', Lambda(la), 'clf', 'lr', 'iwe', p.Results.iwe, 'gamma', p.Results.gamma);
                                
                                % Evaluate on held-out source folds (log-loss)
                                R_la(la) = mean(-yXf.*Xa*theta_f + log(exp(labels(1)*Xa*theta_f) + exp(labels(2)*Xa*theta_f)),1);
                        end
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
            [theta{r,n,m},iw{r,n,m},R(r,n,m),e(r,n,m),pred{r,n,m},post{r,n,m},AUC(r,n,m)] = iwc(X, yX, Z(ixNM,:),'yZ', yZ(ixNM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'lambda', lambda, 'clf', p.Results.clf, 'iwe', p.Results.iwe, 'gamma', p.Results.gamma);
        end
    end
end

% Write results
di = 1; while exist([p.Results.saveName 'results_ssb_iwc_' p.Results.iwe '_'  p.Results.clf '_' num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = [p.Results.saveName 'results_ssb_iwc_' p.Results.iwe '_'  p.Results.clf '_' num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R','iw','e', 'pred', 'post', 'AUC','p');

end
