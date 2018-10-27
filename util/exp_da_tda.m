function exp_da_tda(Z,yZ,varargin)
% Domain adaptation experiment for discriminant analysis on
% target domain.

% Parse arguments
p = inputParser;
addOptional(p, 'clf', 'lda');
addOptional(p, 'NM', []);
addOptional(p, 'nR', 5);
addOptional(p, 'nF', 5);
addOptional(p, 'lambda', []);
addOptional(p, 'prep', {''});
addOptional(p, 'svnm', []);
parse(p, varargin{:});

% Setup for learning curves
M = length(yZ);
if ~isempty(p.Results.NM); lNM = length(p.Results.NM); else; lNM = 1; end

% Normalize data
Z = da_prep(Z', p.Results.prep)';

% Preallocate
theta = cell(p.Results.nR,lNM);
l = cell(p.Results.nR,lNM);
e = NaN(p.Results.nR,lNM);
pred = cell(p.Results.nR,lNM);
post = cell(p.Results.nR,lNM);
AUC = NaN(p.Results.nR,lNM);
for m = 1:lNM
    for r = 1:p.Results.nR
        disp(['Running repeat ' num2str(r) '/' num2str(p.Results.nR)]);
        
        % Select learning curve sample size
        if ~isempty(p.Results.NM)
            ixNM = randsample(1:M, p.Results.NM(m), false);
        else
            ixNM = 1:M;
        end
        
        % Slice out validation set
        ixVa = randsample(1:5, length(ixNM), true);
        V = Z(ixNM(ixVa==1),:);
        yV = yZ(ixNM(ixVa==1));
        T = Z(ixNM(ixVa~=1),:);
        yT = yZ(ixNM(ixVa~=1));
        
        % Cross-validate regularization parameter
        if isempty(p.Results.lambda)
            disp(['Cross-validating for regularization parameter']);
            
            % Set range of regularization parameter
            Lambda = [0 10.^[-6:2:1]];
            ll_la = zeros(1,length(Lambda));
            for la = 1:length(Lambda)
                
                % Split folds
                ixFo = randsample(1:p.Results.nF, size(T,1), true);
                for f = 1:p.Results.nF
                    
                    switch p.Results.clf
                        case 'lda'
                            % Train on included folds
                            theta_f = lda(T(ixFo~=f,:),yT(ixFo~=f),Lambda(la));
                            
                            % Evaluate on held-out fold
                            ll_la(la) = ll_la(la) + sum(sum(ll_lda(theta_f{1}, theta_f{2}, theta_f{3}, T(ixFo==f,:), yT(ixFo==f)),2),1)./sum(ixFo==f);
                        case 'qda'
                            % Train on included folds
                            theta_f = qda(T(ixFo~=f,:),yT(ixFo~=f),Lambda(la));
                            
                            % Evaluate on held-out fold
                            ll_la(la) = ll_la(la) + sum(sum(ll_qda(theta_f{1}, theta_f{2}, theta_f{3}, T(ixFo==f,:), yT(ixFo==f)),2),1)./sum(ixFo==f);
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
                theta{r,m} = lda(T,yT,lambda);
                
                % Evaluate average log-likelihood
                l{r,m} = sum(sum(ll_lda(theta{r,m}{1}, theta{r,m}{2}, theta{r,m}{3}, V, yV),2),1)./size(V,1);
                
                % Error and prediction
                [e(r,m),pred{r,m},post{r,m},AUC(r,m)] = lda_err(theta{r,m}{1},theta{r,m}{2},theta{r,m}{3}, V,yV);
                
            case 'qda'
                % Quadratic Discriminant Analysis
                theta{r,m} = qda(T,yT,lambda);
                
                % Evaluate average log-likelihood
                l{r,m} = sum(sum(ll_qda(theta{r,m}{1}, theta{r,m}{2}, theta{r,m}{3}, V, yV),2),1)./size(V,1);
                
                % Error and prediction
                [e(r,m),pred{r,m},post{r,m},AUC(r,m)] = qda_err(theta{r,m}{1},theta{r,m}{2},theta{r,m}{3}, V,yV);
        end
    end
end

% Write results
di = 1; while exist(['results_t' p.Results.clf '_' p.Results.svnm num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = ['results_t' p.Results.clf '_' p.Results.svnm num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'l', 'e', 'pred', 'post', 'AUC', 'lambda', 'p');

end
