function exp_da_rba(X,yX,Z,yZ,varargin)
% Domain adaptation experiment for Robust Bias-Aware

% Parse arguments
p = inputParser;
addOptional(p, 'NN', []);
addOptional(p, 'NM', []);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'gamma', 1);
addOptional(p, 'lambda', 1);
addOptional(p, 'clip', 1000);
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'prep', {''});
addOptional(p, 'saveName', []);
parse(p, varargin{:});

% Shapes
[N,~] = size(X);
[M,~] = size(Z);
if ~isempty(p.Results.NN); lNN = length(p.Results.NN); else; lNN = 1; end
if ~isempty(p.Results.NM); lNM = length(p.Results.NM); else; lNM = 1; end
labels = unique(yX)';
K = numel(labels);

% Normalize data to prevent
X = da_prep(X', p.Results.prep)';
Z = da_prep(Z', p.Results.prep)';

% Preallocate
theta = cell(p.Results.nR,lNN,lNM);
e = NaN(p.Results.nR,lNN,lNM);
R = NaN(p.Results.nR,lNN,lNM);
AUC = NaN(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
iw = cell(p.Results.nR,lNN,lNM);
for m = 1:lNM
    for n = 1:lNN
        for r = 1:p.Results.nR
            disp(['Running repeat ' num2str(r) '/' num2str(p.Results.nR)]);
            
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
                        
                        % Train on included folds
                        theta_f = rba(X(ixNN(ixFo~=f),:),yX(ixNN(ixFo~=f)),Z(ixNM,:), 'xTol',p.Results.xTol,'maxIter', p.Results.maxIter, 'gamma', p.Results.gamma, 'lambda', Lambda(la), 'clip', p.Results.clip);
                        
                        % Augment held-out fold
                        Xa = [X(ixNN(ixFo==f),:) ones(length(ixNN(ixFo==f)),1)];
                        yXf = yX(ixNN(ixFo==f));
                        
                        % Evaluate on held-out source folds (log-loss)
                        for j = 1:size(Xa,1)
                            [~,yi] = max(yXf(j)==labels,[],2);
                            R_la(la) = R_la(la) + (-Xa(j,:)*theta_f(:,yi) + log(sum(exp(Xa(j,:)*theta_f))));
                        end                        
                        R_la(la) = R_la(la)./size(Xa,1);
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
            [theta{r,n,m},iw{r,n,m},R(r,n,m),e(r,n,m),pred{r,n,m},post{r,n,m},AUC(r,n,m)] = rba(X(ixNN,:),yX(ixNN),Z(ixNM,:),'yZ', yZ(ixNM), 'xTol',p.Results.xTol,'maxIter', p.Results.maxIter, 'gamma', p.Results.gamma, 'lambda', lambda, 'clip', p.Results.clip);
            
        end
    end
end

% Write results
di = 1; while exist([p.Results.saveName 'results_da_rba_' num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = [p.Results.saveName 'results_da_rba_' num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R','e', 'pred', 'post', 'AUC','p','iw');

end
