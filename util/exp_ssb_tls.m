function exp_ssb_tls(Z,yZ,varargin)
% Sample selection bias experiment for least-squares on
% target domain.

% Parse arguments
p = inputParser;
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

% Check for y in {-1,+1}
uy = unique(yZ);
if numel(uy) > 2; error('No multiclass yet'); end
yZ(yZ~=1) = -1;

% Normalize data
Z = da_prep(Z', p.Results.prep)';

% Check for augmentation
if ~all(Z(:,end)==1); Z = [Z ones(size(Z,1),1)]; end

% Preallocate
theta = cell(p.Results.nR,lNM);
R = cell(p.Results.nR,lNM);
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
            Lambda = [0 10.^[-6:2:0]];
            R_la = zeros(1,length(Lambda));
            for la = 1:length(Lambda)
                
                % Split folds
                ixFo = randsample(1:p.Results.nF, size(T,1), true);
                for f = 1:p.Results.nF
                    
                    % Train on included set
                    theta_f = lsq(T(ixFo~=f,:),yT(ixFo~=f),Lambda(la));
                    
                    % Evaluate on validation set
                    R_la(la) = R_la(la) + mean((T(ixFo==f,:)*theta_f - yT(ixFo==f)).^2,1);
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
        theta{r,m} = lsq(T,yT,lambda);
        
        % Evaluate average log-likelihood
        R{r,m} = mean((V*theta{r,m} -yV).^2,1);
        
        % Posteriors
        post{r,m} = exp(V*theta{r,m})./(exp(-V*theta{r,m})+exp( V*theta{r,m}));
        
        % Predictions and error
        pred{r,m} = sign(V*theta{r,m});
        
        % Error
        e(r,m) = mean(pred{r,m} ~= yV);
        
        % AUC
        if numel(unique(yV))==1
            AUC(r,m) = NaN;
        else
            [~,~,~,AUC(r,m)] = perfcurve(yV,post{r,m},+1);
        end
    end
end

% Write results
di = 1; while exist(['results_tls_' p.Results.svnm num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = ['results_tls_' p.Results.svnm num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'R', 'e', 'pred', 'post', 'AUC', 'lambda', 'p');

end
