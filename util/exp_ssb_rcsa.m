function exp_ssb_rcsa(D,y, varargin)
% Sample selection bias experiment for Robust Covariate Shift Adjustment
%
% This experiment script calls code from RobustLearn package from Junfeng Weng's
% Robust Covariate Shift Adjustment approach
% Link: https://webdocs.cs.ualberta.ca/~jwen4/codes/RobustLearning.zip

if isempty(which('robust_learn')); error('Add RobustLearn package to path'); end

% Size of dataset
[M,~] = size(D);

% Parse arguments
p = inputParser;
addOptional(p, 'nN', min(10, round(M./2)));
addOptional(p, 'nM', M);
addOptional(p, 'setDiff', false);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 3);
addOptional(p, 'B', 5);
addOptional(p, 'useGamma', true);
addOptional(p, 'downsampling', false);
addOptional(p, 'lambda', 0);
addOptional(p, 'gamma', 0);
addOptional(p, 'clip', 100);
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'ssb', 'sdw');
addOptional(p, 'viz', false);
addOptional(p, 'saveName', []);
parse(p, varargin{:});

% Check for column vector y
if ~iscolumn(y); y = y'; end

% Labeling
labels = unique(y)';
K = numel(labels);
if K>2; error('Binary classification only'); end
if ~all(labels==[-1 +1]); error('Labels {-1,+1} expected'); end

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
e = NaN(p.Results.nR,lNN,lNM);
R = NaN(p.Results.nR,lNN,lNM);
AUC = NaN(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
alpha = cell(p.Results.nR,lNN,lNM);
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
                    ixf = randsample(1:p.Results.nF, p.Results.nN(n), true);
                    for f = 1:p.Results.nF
                        
                        % Set kernel width to N/5 nearest-neighbor average distance
                        D2 = pdist2(X(ixf~=f,:),X(ixf~=f,:), 'euclidean');
                        D2 = sort(D2, 'ascend');
                        options.sigma = mean(D2(:,ceil(p.Results.nN(n)./5)),1);
                        
                        % Find heuristic gamma
                        options.gamma = 0;
                        if options.useGamma
                            [~,~,~,minErrL] = learn(X(ixf~=f,:), Z(ixnM,:), Z(ixnM,:), yX(ixf~=f), yZ(ixnM), options);
                            KXX = gausskernel(X(ixf~=f,:), X(ixf~=f,:), options.sigma);
                            KXZ = gausskernel(X(ixf~=f,:), Z(ixnM,:), options.sigma);
                            KZZ = gausskernel(Z(ixnM,:), Z(ixnM,:), options.sigma);
                            difMean = mean(KXX(:))+mean(KZZ(:))-2.*mean(KXZ(:));
                            options.gamma = 2*minErrL/difMean;
                        else
                            options.gamma = p.Results.gamma;
                        end
                        
                        % Run robust learner
                        theta_f = robust_learn(X(ixf~=f,:), Z(ixnM,:), Z(ixnM,:), yX(ixf~=f), [], options);
                        
                        % Risk with hinge loss
                        KFX = options.kernel(X(ixf==f,:),X(ixf~=f,:),options.sigma);
                        R_la(la) = mean(max(0, 1 - (KFX*theta_f(1:end-1) + theta_f(end)).*yX(ixf==f)),1);
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
            [theta{r,n,m},alpha{r,n,m},iw{r,n,m}] = robust_learn(X, Z(ixnM,:), Xref, yX, [], options);
            
            % Test kernel
            KZX = options.kernel(Z(ixnM,:),X,options.sigma);
            KZXth = KZX*theta{r,n,m}(1:end-1) + theta{r,n,m}(end);
            
            % Risk with hinge loss
            R(r,n,m) = mean(max(0, 1 - KZXth.*yZ(ixnM)),1);
            
            % Predictions
            pred{r,n,m} = sign(KZXth);
            
            % Errors
            e(r,n,m) = mean(pred{r,n,m}~=yZ(ixnM),1);
            
            % Posteriors
            post{r,n,m} = exp(KZXth)./(exp(-KZXth) + exp(KZXth));
            
            % AUC
            [~,~,~,AUC(r,n,m)] = perfcurve(yZ(ixnM),post{r,n,m},+1);
        end
    end
end

% Write results
di = 1; while exist([p.Results.saveName 'results_ssb_rcsa_' num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = [p.Results.saveName 'results_ssb_rcsa_' num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R', 'e', 'pred', 'post', 'AUC', 'iw', 'alpha', 'p');

end
