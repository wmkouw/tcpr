function exp_da_rcsa(X,yX,Z,yZ,varargin)
% Domain adaptation experiment for Robust Covariate Shift Adjustment

% Parse arguments
p = inputParser;
addOptional(p, 'NN', []);
addOptional(p, 'NM', []);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'B', 5);
addOptional(p, 'useGamma', true);
addOptional(p, 'downsampling', false);
addOptional(p, 'lambda', 0);
addOptional(p, 'gamma', 0);
addOptional(p, 'clip', 100);
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'prep', {''});
addOptional(p, 'saveName', []);
parse(p, varargin{:});

% Shape
[N,D] = size(X);
[M,~] = size(Z);
labels = unique(yX);
K = numel(labels);

% Binary classification only
if K>2; error('RCSA code only supports binary classification'); end

% Setup for learning curves
if ~isempty(p.Results.NN); lNN = length(p.Results.NN); else; lNN = 1; end
if ~isempty(p.Results.NM); lNM = length(p.Results.NM); else; lNM = 1; end

% Normalize data to prevent
X = da_prep(X', p.Results.prep)';
Z = da_prep(Z', p.Results.prep)';

% Options
options.B = p.Results.B;
options.useGamma = p.Results.useGamma;
options.maxIter = p.Results.maxIter;
options.tol = p.Results.xTol;

options.kernel = @linearkernel;
options.learner_sigma = 1;
options.type = 'C';

% Preallocate
theta = cell(p.Results.nR,lNN,lNM);
e = NaN(p.Results.nR,lNN,lNM);
R = NaN(p.Results.nR,lNN,lNM);
AUC = NaN(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
iw = cell(p.Results.nR,lNN,lNM);
alpha = cell(p.Results.nR,lNN,lNM);
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
                    
                    % Set current regularization parameter
                    options.beta = Lambda(la);
                    options.C = 1/(N*Lambda(la));
                    
                    % Split folds
                    ixFo = randsample(1:p.Results.nF, length(ixNN), true);
                    for f = 1:p.Results.nF
                        
                        % Set kernel width to n/5 nn average distance
                        D2 = pdist2(X(ixNN(ixFo~=f),:),X(ixNN(ixFo~=f),:), 'euclidean');
                        options.sigma = mean(D2(:,ceil(N./5)),1);
                        
                        % Find heuristic gamma
                        options.gamma = 0;
                        if options.useGamma
                            [~,~,~,minErrL] = learn(X(ixNN(ixFo~=f),:), Z(ixNM,:), Z(ixNM,:), yX(ixNN(ixFo~=f)), yZ(ixNM), options);
                            KXX = gausskernel(X(ixNN(ixFo~=f),:), X(ixNN(ixFo~=f),:), options.sigma);
                            KXZ = gausskernel(X(ixNN(ixFo~=f),:), Z(ixNM,:), options.sigma);
                            KZZ = gausskernel(Z(ixNM,:), Z(ixNM,:), options.sigma);
                            difMean = sum(sum(KXX))./(M*M)+sum(sum(KZZ))./(M*M)-(2./(M*M))*sum(sum(KXZ));
                            options.gamma = 2*minErrL/difMean;
                        else
                            options.gamma = p.Results.gamma;
                        end
                        
                        % Run robust learner
                        theta_f = robust_learn(X(ixNN(ixFo~=f),:), Z(ixNM,:), Z(ixNM,:), yX(ixNN(ixFo~=f)), [], options);
                        
                        % Evaluate on held-out source folds (hinge-loss)
                        KFX = options.kernel(X(ixNN(ixFo==f),:),X(ixNN(ixFo~=f),:),options.sigma);
                        R_la(la) = mean(max(0, 1 - sign(KFX*theta_f(1:end-1) + theta_f(end)).*yX(ixFo==f)),1);
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
            
            % Reset options
            options.beta = lambda;
            options.C = 1/(N*lambda);
            
            % Set kernel width to n/5 nn average distance
            D2 = pdist2(X(ixNN,:),X(ixNN,:), 'euclidean');
            options.sigma = mean(D2(:,ceil(N./5)),1);
            
            % Find heuristic gamma
            if options.useGamma
                options.gamma = 0;
                Xref = Z(ixNM,:);
                [~,~,~,minErrL] = learn(X(ixNN,:), Z(ixNM,:), Xref, yX(ixNN), yZ(ixNM), options);
                KXX = gausskernel(X(ixNN,:), X(ixNN,:), options.sigma);
                KXZ = gausskernel(X(ixNN,:), Z(ixNM,:), options.sigma);
                KZZ = gausskernel(Z(ixNM,:), Z(ixNM,:), options.sigma);
                difMean = sum(sum(KXX))./(N*N)+sum(sum(KZZ))./(M*M)-(2./(M*N))*sum(sum(KXZ));
                options.gamma = 2*minErrL/difMean;
            else
                Xref = X(ixNN,:);
                options.gamma = p.Results.gamma;
            end
            
            % Learn Robust model (RSCA)
            [theta{r,n,m},alpha{r,n,m},iw{r,n,m}] = robust_learn(X(ixNN,:), Z(ixNM,:), Xref, yX(ixNN), [], options);
            
            % Test kernel
            KZX = options.kernel(Z(ixNM,:),X(ixNN,:),options.sigma);
            KZXth = KZX*theta{r,n,m}(1:end-1) + theta{r,n,m}(end);
            
            % Risk with hinge loss
            R(r,n,m) = mean(max(0, 1 - KZXth.*yZ(ixNM)),1);
            
            % Predictions
            pred{r,n,m} = sign(KZXth);
            
            % Errors
            e(r,n,m) = mean(pred{r,n,m}~=yZ,1);
            
            % Posterior for positive class
            post{r,n,m} = exp(KZXth)./(exp(-KZXth) + exp( KZXth));
            
            % AUC
            [~,~,~,AUC(r,n,m)] = perfcurve(yZ(ixNM),post{r,n,m},+1);
        end
    end
end

% Write results
di = 1; while exist([p.Results.saveName 'results_rcsa_' num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = [p.Results.saveName 'results_rcsa_' num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R', 'e', 'pred', 'post', 'AUC', 'lambda', 'p', 'iw', 'alpha');

end
