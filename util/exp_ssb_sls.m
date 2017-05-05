function exp_ssb_sls(X,yX,Z,yZ,varargin)
% Sample selection bias experiment for least-squares on
% source domain and measure on target domain.

% Parse arguments
p = inputParser;
addOptional(p, 'NN', []);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'lambda', []);
addOptional(p, 'prep', {''});
addOptional(p, 'svnm', []);
addOptional(p, 'tst', []);
parse(p, varargin{:});

% Setup for learning curves
N = length(yX);
if ~isempty(p.Results.NN); lN = length(p.Results.NN); else; lN = 1; end

% Check for y in {-1,+1}
uy = unique(yZ);
if numel(uy) > 2; error('No multiclass yet'); end
yX(yX~=1) = -1;
yZ(yZ~=1) = -1;

% Normalize data
X = da_prep(X', p.Results.prep)';
Z = da_prep(Z', p.Results.prep)';

% Check for augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end
if ~all(Z(:,end)==1); Z = [Z ones(size(Z,1),1)]; end

% Preallocate
theta = cell(p.Results.nR,lN);
R = cell(p.Results.nR,lN);
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
            R_la = zeros(1,length(Lambda));
            for la = 1:length(Lambda)
                
                % Split folds
                ixFo = randsample(1:p.Results.nF, length(ixNN), true);
                for f = 1:p.Results.nF
                    % Train on included set
                    theta_f = lsq(X(ixNN(ixFo~=f),:),yX(ixNN(ixFo~=f)),Lambda(la));
                    
                    % Evaluate on validation set
                    R_la(la) = R_la(la) + mean(X(ixNN(ixFo==f),:)*theta_f - yX(ixNN(ixFo==f)).^2,1);
                    
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
        
        % Least-squares
        theta{r,n} = lsq(X(ixNN,:),yX(ixNN,:),lambda);
        
        % Evaluate average log-likelihood
        R{r,n} = mean((Z*theta{r,n} -yZ).^2,1);
        
        % Posteriors
        post{r,n} = exp(Z*theta{r,n})./(exp(-Z*theta{r,n}) + exp(Z*theta{r,n}));
        
        % Predictions
        pred{r,n} = sign(Z*theta{r,n});
        
        % Error
        e(r,n) = mean(pred{r,n} ~= yZ);
        
        % AUC
        [~,~,~,AUC(r,n)] = perfcurve(yZ,post{r,n},+1);
        
    end
end

% Write results
di = 1; while exist(['results_sls_' p.Results.svnm num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = ['results_sls_' p.Results.svnm num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R', 'e', 'pred', 'post', 'AUC', 'lambda', 'p');

end
