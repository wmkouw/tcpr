function exp_da_tca(X,yX,Z,yZ,varargin)
% Domain adaptation experiment for Transfer Component Analysis

% Parse arguments
p = inputParser;
addOptional(p, 'NN', []);
addOptional(p, 'NM', []);
addOptional(p, 'nC', 10);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'prep', {''});
addOptional(p, 'lambda', 0);
addOptional(p, 'mu', 1);
addOptional(p, 'sigma', 1);
addOptional(p, 'saveName', []);
addOptional(p, 'viz', false);
addOptional(p, 'gif', false);
parse(p, varargin{:});

% Setup for learning curves
[N,~] = size(X);
[M,~] = size(Z);
if ~isempty(p.Results.NN); lNN = length(p.Results.NN); else; lNN = 1; end
if ~isempty(p.Results.NM); lNM = length(p.Results.NM); else; lNM = 1; end

% Labels
labels = unique(yX);
K = numel(labels);

% Normalize data
X = da_prep(X, p.Results.prep);
Z = da_prep(Z, p.Results.prep);

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
        for m = 1:lNM
            
            % Select increasing amount of source and target samples
            if ~isempty(p.Results.NN); ixNN = randsample(1:N, p.Results.NN(n), false); else; ixNN = 1:N; end
            if ~isempty(p.Results.NM); ixNM = randsample(1:M, p.Results.NM(m), false); else; ixNM = 1:M; end
            
            % Cross-validate regularization parameter
            if isempty(p.Results.lambda)
                disp(['Cross-validating for regularization parameter']);
                
                % Set range of regularization parameter
                Lambda = [0 10.^[-6:1:3]];
                R_la = zeros(1,length(Lambda));
                for la = 1:length(Lambda)
                    
                    % Split folds
                    ixFo = randsample(1:p.Results.nF, length(ixNN), true);
                    for f = 1:p.Results.nF
                        
                        % Train on included folds
                        theta_f = tca(X(ixNN(ixFo~=f),:),yX(ixNN(ixFo~=f)),Z(ixNM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'nC', p.Results.nC, 'mu', p.Results.mu, 'l2', Lambda(la), 'sigma', p.Results.sigma);
                        
                        % Evaluate on held-out source folds (error)
                        lf = sum(ixFo==f);
                        [Pf,KXZ] = tc(X(ixNN(ixFo==f),:)', Z(ixNM,:)', 'sigma', p.Results.sigma, 'mu', p.Results.mu, 'nC', p.Results.nC);
                        [~,pred_f] = max([KXZ(1:lf,:)*Pf ones(lf,1)]*theta_f, [], 2);
                        R_la(la) = mean(labels(pred_f) ~= yX(ixNN(ixFo==f)),1);
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
            [theta{r,n,m},P{r,n,m},R(r,n,m),e(r,n,m),pred{r,n,m},post{r,n,m},AUC(r,n,m)] = tca(X(ixNN,:), yX(ixNN), Z(ixNM,:),'yZ', yZ(ixNM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'nC', p.Results.nC, 'mu', p.Results.mu, 'sigma', p.Results.sigma, 'l2', lambda);
        end
    end
end

% Write results
di = 1; while exist([p.Results.saveName 'results_da_tca_' num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = [p.Results.saveName 'results_da_tca_' num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R', 'e', 'post', 'AUC', 'lambda', 'p', 'P');

end
