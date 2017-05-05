function exp_ssb_rcsa(D,y, varargin)
% Sample selection bias experiment for Robust Covariate Shift Adjustment

addpath(genpath('../util/minFunc'))
addpath(genpath('../util/RobustLearning'))

% Size of dataset
[M,~] = size(D);

% Parse arguments
p = inputParser;
addOptional(p, 'nN', min(10, round(M./2)));
addOptional(p, 'nM', M);
addOptional(p, 'setDiff', false);
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
addOptional(p, 'svnm', []);
parse(p, varargin{:});

% Normalize data
D = da_prep(D, p.Results.prep);

% Force labels in {-1,+1}
lab = unique(y);
if ~isempty(setdiff(lab,[-1 1]))
    disp(['Forcing labels into {-1,+1}']);
    y(y~=1) = -1;
end

% Number of sample sizes
lNN = length(p.Results.nN);
if isempty(p.Results.nM)
    lNM = 1;
else
    lNM = length(p.Results.nM);
end

% Options
options.B = p.Results.B;
options.useGamma = p.Results.useGamma;
options.maxIter = p.Results.maxIter;
options.tol = p.Results.xTol;

options.kernel = @linearkernel;
options.learner_sigma = 1;
options.type = 'C';

% Preallocation
theta = cell(p.Results.nR,lNN,lNM);
alpha = cell(p.Results.nR,lNN,lNM);
w = cell(p.Results.nR,lNN,lNM);
e = cell(p.Results.nR,lNN,lNM);
R = cell(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
AUC = cell(p.Results.nR,lNN,lNM);


for r = 1:p.Results.nR
    disp(['Running repeat ' num2str(r) '/' num2str(p.Results.nR)]);
    
    for n = 1:lNN
        
        % Select samples
        ix = ssb_nhg(D,y,p.Results.nN(n), 'loc', 'edge', 'type', 'Gaussian', 'viz', false);
        
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
                ixnM = randsample(1:M, p.Results.nM(m), false);
            else
                ixnM = 1:size(Z,1);
            end
            
            % Cross-validate regularization parameter
            if isempty(p.Results.lambda)
                disp(['Cross-validating for regularization parameter']);
                
                % Set range of regularization parameter
                Lambda = [0 10.^[-6:2:0]];
                R_la = zeros(1,length(Lambda));
                for la = 1:length(Lambda)
                    
                    % Set current regularization parameter
                    options.beta = Lambda(la);
                    options.C = 1/(p.Results.nN(n)*Lambda(la));
                    
                    % Split folds
                    ixFo = randsample(1:p.Results.nF, length(ixNN), true);
                    for f = 1:p.Results.nF
                        
                        % Set kernel width to N/5 nearest-neighbor average distance
                        D2 = pdist2(X(ixFo~=f,:),X(ixFo~=f,:), 'euclidean');
                        D2 = sort(D2, 'ascend');
                        options.sigma = mean(D2(:,ceil(p.Results.nN(n)./5)),1);
                        
                        % Find heuristic gamma
                        options.gamma = 0;
                        if options.useGamma
                            [~,~,~,minErrL] = learn(X(ixFo~=f,:), Z(ixnM,:), Z(ixnM,:), yX(ixFo~=f), yZ(ixnM), options);
                            KXX = gausskernel(X(ixFo~=f,:), X(ixFo~=f,:), options.sigma);
                            KXZ = gausskernel(X(ixFo~=f,:), Z(ixnM,:), options.sigma);
                            KZZ = gausskernel(Z(ixnM,:), Z(ixnM,:), options.sigma);
                            difMean = mean(KXX(:))+mean(KZZ(:))-2.*mean(KXZ(:));
                            options.gamma = 2*minErrL/difMean;
                        else
                            options.gamma = p.Results.gamma;
                        end
                        
                        % Run robust learner
                        theta_f = robust_learn(X(ixFo~=f,:), Z(ixnM,:), Z(ixnM,:), yX(ixFo~=f), [], options);
                        
                        % Risk with hinge loss
                        KFX = options.kernel(X(ixFo==f,:),X(ixFo~=f,:),options.sigma);
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
            options.C = 1/(p.Results.nN(n)*lambda);
            
            % Set kernel width to N/5 nearest-neighbor average distance
            D2 = pdist2(X,X, 'euclidean');
            D2 = sort(D2,'ascend');
            options.sigma = mean(D2(:,ceil(p.Results.nN(n)./5)),1);
            
            % Find heuristic gamma
            if options.useGamma
                options.gamma = 0;
                Xref = Z(ixnM,:);
                [~,~,~,minErrL] = learn(X, Z(ixnM,:), Xref, yX, yZ(ixnM), options);
                KXX = gausskernel(X, X, options.sigma);
                KXZ = gausskernel(X, Z(ixnM,:), options.sigma);
                KZZ = gausskernel(Z(ixnM,:), Z(ixnM,:), options.sigma);
                difMean = mean(KXX(:))+mean(KZZ(:))-2.*mean(KXZ(:));
                options.gamma = 2*minErrL/difMean;
            else
                Xref = X;
                options.gamma = p.Results.gamma;
            end
            
            % Learn Robust model (RSCA)
            [theta{r,n,m},alpha{r,n,m},w{r,n,m}] = robust_learn(X, Z(ixnM,:), Xref, yX, [], options);
            
            % Test kernel
            KZX = options.kernel(Z(ixnM,:),X,options.sigma);
            KZXth = KZX*theta{r,n,m}(1:end-1) + theta{r,n,m}(end);
            
            % Predictions
            pred{r,n,m} = sign(KZXth);
            
            % Risk with hinge loss
            R{r,n,m} = mean(max(0, 1 - pred{r,n,m}.*yZ(ixnM)),1);
            
            % Errors
            e{r,n,m} = mean(pred{r,n,m}~=yZ(ixnM),1);
            
            % Posteriors
            post{r,n,m} = exp(KZXth)./(1 + exp(KZXth));
            
            % AUC
            [~,~,~,AUC{r,n,m}] = perfcurve(yZ(ixnM),post{r,n,m},1);
        end
    end
end

% Write results
di = 1; while exist(['results_rcsa_' p.Results.svnm num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = ['results_rcsa_' p.Results.svnm num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R', 'e', 'pred', 'post', 'AUC', 'w', 'alpha', 'p');

end
