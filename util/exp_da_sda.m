function exp_da_sda(X,yX,Z,yZ,varargin)
% Domain adaptation experiment for TCP

% Parse arguments
p = inputParser;
addOptional(p, 'clf', 'lda');
addOptional(p, 'NN', []);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'lambda', []);
addOptional(p, 'prep', {''});
addOptional(p, 'saveName', []);
addOptional(p, 'tst', []);
parse(p, varargin{:});

% Setup for learning curves
N = length(yX);
if ~isempty(p.Results.NN)
    lN = length(p.Results.NN);
else
    lN = 1;
end

% Normalize data
X = da_prep(X', p.Results.prep)';
Z = da_prep(Z', p.Results.prep)';

% Preallocate
theta = cell(p.Results.nR,lN);
l = cell(p.Results.nR,lN);
e = NaN(p.Results.nR,lN);
pred = cell(p.Results.nR,lN);
post = cell(p.Results.nR,lN);
AUC = NaN(p.Results.nR,lN);
for n = 1:lN
    for r = 1:p.Results.nR
        disp(['Running repeat ' num2str(r) '/' num2str(p.Results.nR)]);
        
        % Select increasing amount of unlabeled samples
        if ~isempty(p.Results.NN)
            ixNN = randsample(1:N, p.Results.NN(n), false);
        else
            ixNN = 1:N;
        end
        
        % Cross-validate regularization parameter
        if isempty(p.Results.lambda)
            disp(['Cross-validating for regularization parameter']);
            
            % Set range of regularization parameter
            Lambda = [0 10.^[-6:2:0]];
            ll_la = zeros(1,length(Lambda));
            for la = 1:length(Lambda)
                
                % Split folds
                ixFo = randsample(1:p.Results.nF, length(ixNN), true);
                for f = 1:p.Results.nF
                    
                    switch p.Results.clf
                        case 'lda'
                            % Train
                            theta_f = lda(X(ixNN(ixFo~=f),:),yX(ixNN(ixFo~=f)),Lambda(la));
                            
                            % Evaluate on validation set
                            ll_la(la) = ll_la(la) + sum(sum(ll_lda(theta_f{1}, theta_f{2}, theta_f{3}, X(ixNN(ixFo==f),:), yX(ixNN(ixFo==f))),2),1)./sum(ixFo==f);
                        case 'qda'
                            % Train
                            theta_f = qda(X(ixNN(ixFo~=f),:),yX(ixNN(ixFo~=f)),Lambda(la));
                            
                            % Evaluate on validation set
                            ll_la(la) = ll_la(la) + sum(sum(ll_qda(theta_f{1}, theta_f{2}, theta_f{3}, X(ixNN(ixFo==f),:), yX(ixNN(ixFo==f))),2),1)./sum(ixFo==f);
                    end
                end
            end
            % Select minimal 
            ll_la(isinf(ll_la)) = NaN;
            [~,ixMinLambda] = max(ll_la);
            lambda = Lambda(ixMinLambda);
        else
            lambda = p.Results.lambda;
        end
        disp(['\lambda = ' num2str(lambda)]);
        
        switch p.Results.clf
            case 'lda'
                % Linear Discriminant Analysis
                theta{r,n} = lda(X(ixNN,:),yX(ixNN,:),lambda);
                
                % Evaluate average log-likelihood
                l{r,n} = sum(sum(ll_lda(theta{r,n}{1}, theta{r,n}{2}, theta{r,n}{3}, Z, yZ),2),1)./size(Z,1);
                
                % Error and prediction
                [e(r,n),pred{r,n},post{r,n},AUC(r,n)] = lda_err(theta{r,n}{1},theta{r,n}{2},theta{r,n}{3}, Z,yZ);
                
            case 'qda'
                % Quadratic Discriminant Analysis
                theta{r,n} = qda(X(ixNN,:),yX(ixNN,:),lambda);
                
                % Evaluate average log-likelihood
                l{r,n} = sum(sum(ll_qda(theta{r,n}{1}, theta{r,n}{2}, theta{r,n}{3}, Z, yZ),2),1)./size(Z,1);
                
                % Error and prediction
                [e(r,n),pred{r,n},post{r,n},AUC(r,n)] = qda_err(theta{r,n}{1},theta{r,n}{2},theta{r,n}{3}, Z,yZ);
        end
    end
end

% Write results
di = 1; while exist([p.Results.saveName 'results_s' p.Results.clf '_' num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = [p.Results.saveName 'results_s' p.Results.clf '_' num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'l', 'e', 'pred', 'post', 'AUC', 'lambda', 'p');

end
