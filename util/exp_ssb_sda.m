function exp_ssb_sda(D,y,varargin)
% Sample selection bias experiment for discriminant analysis on
% source domain and measure on target domain.

% Size of dataset
[M,~] = size(D);

% Parse arguments
p = inputParser;
addOptional(p, 'clf', 'lda');
addOptional(p, 'nN', min(10, round(M./2)));
addOptional(p, 'nM', M);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'lambda', []);
addOptional(p, 'prep', {''});
addOptional(p, 'svnm', []);
addOptional(p, 'tst', []);
addOptional(p, 'setDiff', false);
parse(p, varargin{:});

% Normalize data
D = da_prep(D, p.Results.prep);

% Number of sample sizes
lNN = length(p.Results.nN);
if isempty(p.Results.nM)
    lNM = 1;
else
    lNM = length(p.Results.nM);
end

% Preallocate
theta = cell(p.Results.nR,lNN,lNM);
e = zeros(p.Results.nR,lNN,lNM);
R = cell(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
AUC = zeros(p.Results.nR,lNN,lNM);

for r = 1:p.Results.nR
    disp(['Running repeat ' num2str(r) '/' num2str(p.Results.nR)]);
    
    for n = 1:lNN
        
        % Select samples
        [X,yX,Z,yZ] = ssb_nn(D,y, 'N', p.Results.nN(n), 'setDiff', p.Results.setDiff);
        
        for m = 1:lNM
            
            % Select target samples
            if ~isempty(p.Results.nM)
                ixnM = randsample(1:M, p.Results.nM(m), false);
            else
                ixnM = 1:size(Z,1);
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
                                theta_f = lda(X(ixFo~=f,:),yX(ixFo~=f),Lambda(la));
                                
                                % Evaluate on validation set
                                ll_la(la) = ll_la(la) + sum(sum(ll_lda(theta_f{1}, theta_f{2}, theta_f{3}, X(ixFo==f,:), yX(ixFo==f)),2),1)./sum(ixFo==f);
                            case 'qda'
                                % Train
                                theta_f = qda(X(ixFo~=f,:),yX(ixFo~=f),Lambda(la));
                                
                                % Evaluate on validation set
                                ll_la(la) = ll_la(la) + sum(sum(ll_qda(theta_f{1}, theta_f{2}, theta_f{3}, X(ixFo==f,:), yX(ixFo==f)),2),1)./sum(ixFo==f);
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
                    theta{r,n,m} = lda(X,yX,lambda);
                    
                    % Evaluate average log-likelihood
                    R{r,n,m} = sum(sum(ll_lda(theta{r,n,m}{1}, theta{r,n,m}{2}, theta{r,n,m}{3}, Z(ixnM,:), yZ(ixnM)),2),1)./size(Z(ixnM,:),1);
                    
                    % Error and prediction
                    [e(r,n,m),pred{r,n,m},post{r,n,m},AUC(r,n,m)] = lda_err(theta{r,n,m}{1},theta{r,n,m}{2},theta{r,n,m}{3}, Z(ixnM,:),yZ(ixnM));
                    
                case 'qda'
                    % Quadratic Discriminant Analysis
                    theta{r,n,m} = qda(X,yX,lambda);
                    
                    % Evaluate average log-likelihood
                    R{r,n,m} = sum(sum(ll_qda(theta{r,n,m}{1}, theta{r,n,m}{2}, theta{r,n,m}{3}, Z(ixnM,:), yZ(ixnM)),2),1)./size(Z(ixnM,:),1);
                    
                    % Error and prediction
                    [e(r,n,m),pred{r,n,m},post{r,n,m},AUC(r,n,m)] = qda_err(theta{r,n,m}{1},theta{r,n,m}{2},theta{r,n,m}{3}, Z(ixnM,:),yZ(ixnM));
            end
        end
    end
end

% Write results
di = 1; while exist(['results_s' p.Results.clf '_' p.Results.svnm num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = ['results_s' p.Results.clf '_' p.Results.svnm num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R', 'e', 'pred', 'post', 'AUC', 'lambda', 'p');

end
