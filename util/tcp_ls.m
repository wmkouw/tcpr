function [theta,varargout] = tce_ls(X,yX,Z,varargin)
% Function to run the Least-Squares version of the Target Contrastive Pessimistic Estimator
% Input:
% 		    X      	source data (N samples x D features)
%           Z      	target data (M samples x D features)
%           yX 	   	source labels (N x 1)
% Optional input:
%     		yZ 		target labels (M samples x 1, for evaluation)
% 			alpha 	learning rate accelerant (default: 1)
%           lambda  l2-regularization parameter (default: 0)
% 			maxIter maximum number of iterations (default: 500)
% 			xTol 	convergence criterion (default: 1e-5)
% Output:
% 			theta   tce estimate
% Optional output:
%           {1}     found worst-case labeling q
%           {2}   	target loss of the mcpl/ref estimate with q/u
% 			{3} 	target err/pred of the mcpl/ref esimate with u
%
% Wouter M. Kouw (2017). Target Contrastive Pessimistic Risk
% Last update: 27-01-2017

% Parse hyperparameters
p = inputParser;
addOptional(p, 'yZ', []);
addOptional(p, 'alpha', 2);
addOptional(p, 'lambda', 0);
addOptional(p, 'maxIter', 5e4);
addOptional(p, 'xTol', 1e-12);
addOptional(p, 'lr', 'geom');
parse(p, varargin{:});

% Augment data with bias if necessary
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end
if ~all(Z(:,end)==1); Z = [Z ones(size(Z,1),1)]; end

% Size
[N,D] = size(X);
[M,~] = size(Z);
lab = unique(yX);
K = numel(lab);
if K~=2; error('Binary classification only'); end

% Force source labels in {-1,+1}
if ~isempty(setdiff(lab,[-1 1]))
    disp(['Forcing labels into {-1,+1}']);
    yX(yX~=1) = -1;
    lab = [-1 1];
end

% Reference parameter estimates
theta.ref = (X'*X + p.Results.lambda*eye(D))\(X'*yX);

% Initialize
q = ones(M,K)./K;
Dq = zeros(M,K);
theta.tcp = theta.ref;

Rmm = Inf;
disp('Starting MCPL optimization');
for n = 1:p.Results.maxIter
    
    %%% Minimization
    
    % Closed-form minimization w.r.t. theta
    ZZ = svdinv(Z'*Z + p.Results.lambda*eye(D));
    [~,psd] = chol(ZZ); if psd>0; disp('target covariance not psd'); end
    theta.tcp = ZZ*(Z'*(lab(1)*q(:,1) + lab(2)*q(:,2)));
    
    %%% Maximization
    
    % Compute new gradient
    for k = 1:K
        Dq(:,k) = (Z*theta.tcp - lab(k)).^2 - (Z*theta.ref - lab(k)).^2;
    end
    
    % Update learning rate
    switch p.Results.lr
        case 'lin'
            lr = (p.Results.maxIter - n)./(p.Results.alpha*p.Results.maxIter);
        case 'quad'
            lr = ((p.Results.maxIter - n)./(p.Results.alpha*p.Results.maxIter)).^2;
        case 'geom'
            lr = 1./(p.Results.alpha*n);
        case 'exp'
            lr = exp(-p.Results.alpha*n);
    end
    
    % Apply gradient
    q = q + lr.*Dq;
    
    % Project back onto simplex
    q = myprojsplx(q);
    
    %%% Check progress
    
    % Risk of mcpl and ref estimates
    R_tcp = mean(q(:,1).*(Z*theta.tcp - lab(1)).^2 + q(:,2).*(Z*theta.tcp - lab(2)).^2,1);
    R_ref = mean(q(:,1).*(Z*theta.ref - lab(1)).^2 + q(:,2).*(Z*theta.ref - lab(2)).^2,1);
    
    % Check for update under tolerance
    Rmm_ = R_tcp - R_ref;
    dR = norm(Rmm-Rmm_);
    if  (dR < p.Results.xTol) && (n > 5)  || isnan(Rmm_)
        disp(['Broke at ' num2str(n) ' with loss ' num2str(Rmm_)]);
        break;
    end
    
    % Report progress
    if rem(n,100)==1
        disp(['Iteration ' num2str(n) '/' num2str(p.Results.maxIter) ' - Minimax loss: ' num2str(Rmm_)]);
    end
    
    % Update likelihood
    Rmm = Rmm_;
end

% Oracle parameter estimates
theta.orc = (Z'*Z + p.Results.lambda*eye(D))\(Z'*p.Results.yZ);

%%% Optional output
if nargout > 1
    
    % Risk of found worst-case labeling
    R.tcp_q = mean(q(:,1).*(Z*theta.tcp - lab(1)).^2 + q(:,2).*(Z*theta.tcp - lab(2)).^2,1);
    R.ref_q = mean(q(:,1).*(Z*theta.ref - lab(1)).^2 + q(:,2).*(Z*theta.ref - lab(2)).^2,1);
    R.orc_q = mean(q(:,1).*(Z*theta.orc - lab(1)).^2 + q(:,2).*(Z*theta.orc - lab(2)).^2,1);
    
    if ~isempty(p.Results.yZ)
        
        % Force target labels in {-1,+1}
        yZ = p.Results.yZ; yZ(yZ~=1) = -1;
        
        % Risk of true labeling
        R.tcp_u = mean((Z*theta.tcp - yZ).^2,1);
        R.ref_u = mean((Z*theta.ref - yZ).^2,1);
        R.orc_u = mean((Z*theta.orc - yZ).^2,1);
        
        % Posteriors
        post.tcp = exp(Z*theta.tcp)./(exp(-Z*theta.tcp) + exp(Z*theta.tcp));
        post.ref = exp(Z*theta.ref)./(exp(-Z*theta.ref) + exp(Z*theta.ref));
        post.orc = exp(Z*theta.orc)./(exp(-Z*theta.orc) + exp(Z*theta.orc));
        
        % Predictions
        pred.tcp = sign(Z*theta.tcp);
        pred.ref = sign(Z*theta.ref);
        pred.orc = sign(Z*theta.orc);
        
        % Error on true labeling
        e.tcp = mean(pred.tcp ~= yZ);
        e.ref = mean(pred.ref ~= yZ);
        e.orc = mean(pred.orc ~= yZ);
        
        % AUC on true labeling
        [~,~,~,AUC.tcp] = perfcurve(yZ,post.tcp,+1);
        [~,~,~,AUC.ref] = perfcurve(yZ,post.ref,+1);
        [~,~,~,AUC.orc] = perfcurve(yZ,post.orc,+1);
        
        % Output predictions and error
        varargout{3} = e;
        varargout{4} = pred;
        varargout{5} = post;
        varargout{6} = AUC;
    end
    
    % Output
    varargout{1} = q;
    varargout{2} = R;
    
end

end
