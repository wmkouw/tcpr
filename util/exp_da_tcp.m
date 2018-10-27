function exp_da_tcp(X,yX,Z,yZ,varargin)
% Domain adaptation experiment for TCP

% Parse arguments
p = inputParser;
addOptional(p, 'clf', 'tcp-lda');
addOptional(p, 'NN', []);
addOptional(p, 'NM', []);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'lr', 'geom');
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'prep', {''});
addOptional(p, 'lambda', 0);
addOptional(p, 'alpha', 1);
addOptional(p, 'saveName', []);
addOptional(p, 'viz', false);
addOptional(p, 'gif', false);
parse(p, varargin{:});

% Setup for learning curves
[N,~] = size(X);
[M,~] = size(Z);
if ~isempty(p.Results.NN); lNN = length(p.Results.NN); else; lNN = 1; end
if ~isempty(p.Results.NM); lNM = length(p.Results.NM); else; lNM = 1; end

% Normalize data
X = da_prep(X, p.Results.prep);
Z = da_prep(Z, p.Results.prep);

% Preallocate
theta = cell(p.Results.nR,lNN,lNM);
e = cell(p.Results.nR,lNN,lNM);
R = cell(p.Results.nR,lNN,lNM);
AUC = cell(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
q = cell(p.Results.nR,lNN,lNM);
for r = 1:p.Results.nR
    
    if (rem(r, p.Results.nR./10)==1)
        fprintf('At repetition \t%i/%i\n', r, nR)
    end
    
    for n = 1:lNN
        for m = 1:lNM
            
            % Select increasing amount of source and target samples
            if ~isempty(p.Results.NN); ixNN = randsample(1:N, p.Results.NN(n), false); else; ixNN = 1:N; end
            if ~isempty(p.Results.NM); ixNM = randsample(1:M, p.Results.NM(m), false); else; ixNM = 1:M; end
            
            % Cross-validate regularization parameter
            if isempty(p.Results.lambda)
                disp(['Cross-validating for regularization parameter']);
                
                % Set range of regularization parameter
                Lambda = [10.^[-6:1:3]];
                R_la = zeros(1,length(Lambda));
                for la = 1:length(Lambda)
                    
                    % Split folds
                    ixFo = randsample(1:p.Results.nF, length(ixNN), true);
                    for f = 1:p.Results.nF
                        
                        switch p.Results.clf
                            case 'tcp-ls'
                                % Train on included folds
                                theta_f = tcp_ls(X(ixNN(ixFo~=f),:),yX(ixNN(ixFo~=f)),Z(ixNM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', Lambda(la), 'lr', p.Results.lr);
                                
                                % Evaluate on held-out source folds (MSE)
                                R_la(la) = mean(([X(ixNN(ixFo==f),:) ones(length(ixNN(ixFo==f)),1)]*theta_f.tcp - yX(ixNN(ixFo==f))).^2);
                            case 'tcp-lda'
                                % Train on included folds
                                theta_f = tcp_lda(X(ixNN(ixFo~=f),:),yX(ixNN(ixFo~=f)),Z(ixNM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', Lambda(la), 'lr', p.Results.lr);
                                
                                % Evaluate on held-out source folds (-ALL)
                                R_la(la) = R_la(la) - sum(sum(ll_lda(theta_f.tcp{1}, theta_f.tcp{2}, theta_f.tcp{3}, X(ixNN(ixFo==f),:), yX(ixNN(ixFo==f))),2),1)./sum(ixNN(ixFo==f));
                            case 'tcp-qda'
                                % Train on included folds
                                theta_f = tcp_qda(X(ixNN(ixFo~=f),:),yX(ixNN(ixFo~=f)),Z(ixNM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', Lambda(la), 'lr', p.Results.lr);
                                
                                % Evaluate on held-out source folds (-ALL)
                                R_la(la) = R_la(la) - sum(sum(ll_qda(theta_f.tcp{1}, theta_f.tcp{2}, theta_f.tcp{3}, X(ixNN(ixFo==f),:), yX(ixNN(ixFo==f))),2),1)./sum(ixNN(ixFo==f));
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
            switch p.Results.clf
                case 'tcp-ls'
                    [theta{r,n,m},q{r,n,m},R{r,n,m},e{r,n,m},pred{r,n,m},post{r,n,m},AUC{r,n,m}] = tcp_ls(X(ixNN,:), yX(ixNN), Z(ixNM,:),'yZ', yZ(ixNM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', lambda, 'lr', p.Results.lr);
                case 'tcp-lda'
                    [theta{r,n,m},q{r,n,m},R{r,n,m},e{r,n,m},pred{r,n,m},post{r,n,m},AUC{r,n,m}] = tcp_lda(X(ixNN,:), yX(ixNN), Z(ixNM,:),'yZ', yZ(ixNM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', lambda, 'lr', p.Results.lr);
                case 'tcp-qda'
                    [theta{r,n,m},q{r,n,m},R{r,n,m},e{r,n,m},pred{r,n,m},post{r,n,m},AUC{r,n,m}] = tcp_qda(X(ixNN,:), yX(ixNN), Z(ixNM,:),'yZ', yZ(ixNM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', lambda, 'lr', p.Results.lr);
            end
        end
    end
end

% Write results
di = 1; while exist([p.Results.saveName 'results_da_' p.Results.clf '_' num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = [p.Results.saveName 'results_da_' p.Results.clf '_' num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R', 'e', 'post', 'AUC', 'lambda', 'p', 'q');

end
