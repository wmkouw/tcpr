function exp_da_iwc(X,yX,Z,yZ,varargin)
% Domain adaptation experiment for importance weighted
% classifiers on source domain and measure on target domain.

% Parse arguments
p = inputParser;
addOptional(p, 'clf', 'lsq');
addOptional(p, 'NN', []);
addOptional(p, 'NM', []);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'lambda', 1);
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'svnm', []);
addOptional(p, 'iwe', 'kmm');
addOptional(p, 'prep', {''});
parse(p, varargin{:});

% Setup for learning curves
[N,~] = size(X);
[M,~] = size(Z);
if ~isempty(p.Results.NN); lNN = length(p.Results.NN); else; lNN = 1; end
if ~isempty(p.Results.NM); lNM = length(p.Results.NM); else; lNM = 1; end

% Normalize data to prevent
X = da_prep(X', p.Results.prep)';
Z = da_prep(Z', p.Results.prep)';

switch p.Results.clf
    case 'lsq'
        % Force labels in {-1,+1}
        lab = union(unique(yX),unique(yZ));
        if ~isempty(setdiff(lab,[-1 1]))
            disp(['Forcing labels into {-1,+1}']);
            yX(yX~=1) = -1;
            yZ(yZ~=1) = -1;
        end
    case 'lr'
        % Force labels in {-1,+1}
        lab = union(unique(yX),unique(yZ));
        if ~isempty(setdiff(lab,[0 1]))
            disp(['Forcing labels into {0,+1}']);
            yX(yX~=1) = 0;
            yZ(yZ~=1) = 0;
        end
end

% Preallocate
theta = cell(p.Results.nR,lNN,lNM);
e = NaN(p.Results.nR,lNN,lNM);
R = NaN(p.Results.nR,lNN,lNM);
AUC = NaN(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
w = cell(p.Results.nR,lNN,lNM);
for r = 1:p.Results.nR
    disp(['Running repeat ' num2str(r) '/' num2str(p.Results.nR)]);
    
    for n = 1:lNN
        for m = 1:lNM
            
            % Select increasing amount of source and target samples
            if ~isempty(p.Results.NN); ixNN = randsample(1:N, p.Results.NN(n), false); else; ixNN = 1:N; end
            if ~isempty(p.Results.NM); ixNM = randsample(1:M, p.Results.NM(m), false); else; ixNM = 1:M; end
            
            % Cross-validate regularization parameter
            if isempty(p.Results.lambda)
                disp(['Cross-validating for regularization parameter']);
                
                % Set range of regularization parameter
                Lambda = [0 10.^[-6:2:0]];
                R_la = zeros(1,length(Lambda));
                for la = 1:length(Lambda)
                    
                    % Split folds
                    ixFo = randsample(1:p.Results.nF, length(ixNN), true);
                    for f = 1:p.Results.nF
                        
                        % Augment held-out fold
                        Xa = [X(ixNN(ixFo==f),:) ones(length(ixNN(ixFo==f)),1)];
                        yXf = yX(ixNN(ixFo==f));
                        
                        switch p.Results.clf
                            case 'lsq'
                                % Train on included folds
                                theta_f = iwc(X(ixNN(ixFo~=f),:),yX(ixNN(ixFo~=f)),Z(ixNM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'lambda', Lambda(la), 'clf', 'lsq', 'iwe', p.Results.iwe);
                                
                                % Evaluate on held-out source folds (MSE)
                                R_la(la) = mean((Xa*theta_f - yXf).^2,1);
                            case 'lr'
                                % Train on included folds
                                [theta_f,w] = iwc(X(ixNN(ixFo~=f),:),yX(ixNN(ixFo~=f)),Z(ixNM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'lambda', Lambda(la), 'clf', 'lr', 'iwe', p.Results.iwe);
                                
                                % Evaluate on held-out source folds (log-loss)
                                R_la(la) = mean(-yXf.*Xa*theta_f + log(1 + exp(Xa*theta_f)),1);
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
            [theta{r,n,m},w{r,n,m},R(r,n,m),e(r,n,m),pred{r,n,m},post{r,n,m},AUC(r,n,m)] = iwc(X(ixNN,:), yX(ixNN), Z(ixNM,:),'yZ', yZ(ixNM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'lambda', lambda, 'clf', p.Results.clf, 'iwe', p.Results.iwe);
        end
    end
end

% Write results
di = 1; while exist(['results_iwc_' p.Results.iwe '_' p.Results.clf '_' p.Results.svnm num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = ['results_iwc_' p.Results.iwe '_' p.Results.clf '_' p.Results.svnm num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R', 'e', 'post', 'AUC', 'lambda', 'p', 'w');

end
