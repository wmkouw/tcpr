function [theta,varargout] = tcp_lda(X,yX,Z,varargin)
% Linear Discriminant Analysis version of the Target Contrastive Pessimistic Estimator
% Input:
% 		    X      	source data (N samples x D features)
%           Z      	target data (M samples x D features)
%           yX 	   	source labels (N x 1)
% Optional input:
%     		yZ 		target labels (M samples x 1, for evaluation)
%           lr      learning rate (default: 'geom' = geometric)
% 			alpha 	learning rate (default: 1)
%           lambda  l2-regularization parameter (default: 0)
% 			maxIter maximum number of iterations (default: 500)
% 			xTol 	convergence criterion (default: 1e-5)
% Output:
% 			theta   target model estimate
% Optional output:
%           {1}     found worst-case labeling q
%           {2}   	target risks
% 			{3} 	target errors
%           {4}     target predictions
%           {5}     target posteriors
%           {6}     target area under the ROC-curve
%
% Wouter M. Kouw (2017). Target Contrastive Pessmistic Risk.
% Last update: 19-05-2017

% Parse hyperparameters
p = inputParser;
addOptional(p, 'yZ', []);
addOptional(p, 'lr', 'geom');
addOptional(p, 'alpha', 2);
addOptional(p, 'lambda', 0);
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
parse(p, varargin{:});

% Sizes
[N,D] = size(X);
[M,~] = size(Z);

% Check for column vector y
if ~iscolumn(yX); yX = yX'; end

% Labeling
labels = unique(yX)';
K = numel(labels);

% Preallocation
if K==1
    pi_ref = zeros(1,2);
    mu_ref = zeros(2,D);
    Si_ref = zeros(D,D,2);
else
    pi_ref = zeros(1,K);
    mu_ref = zeros(K,D);
    Si_ref = zeros(D,D,K);
end

% Reference parameter estimates
for k = 1:K
    
    % Parameters
    Nk = sum(yX==labels(k));
    pi_ref(k) = Nk./N;
    mu_ref(k,:) = sum(X(yX==labels(k),:),1)./Nk;
    Si_ref(:,:,k) = (bsxfun(@minus,X(yX==labels(k),:),mu_ref(k,:))'*bsxfun(@minus,X(yX==labels(k),:),mu_ref(k,:)))./Nk;
    
    % Regularization
    if p.Results.lambda>0
        % Constrain covariance to minimum size
        [V,E] = eig((Si_ref(:,:,k)+Si_ref(:,:,k)')./2);
        E = max(0, E - p.Results.lambda.*eye(D));
        Si_ref(:,:,k) = V*E*V' + p.Results.lambda*eye(D);
    end
end

% Combine class-covariances
for k = 1:K
    Si_ref(:,:,k) = Si_ref(:,:,k)*pi_ref(k);
end
Si_ref = sum(Si_ref,3);

% Log-likelihood of unlabeled samples under reference model
R_ref = ll_lda(pi_ref,mu_ref,Si_ref,Z);

% Initialize target posterior
q = ones(M,K)./K;

% Start optimization
Rmm = Inf;
disp('Starting TCP optimization');
for n = 1:p.Results.maxIter
    
    %%% Maximization
    if K==1
        pi_tcp = zeros(1,2);
        mu_tcp = zeros(2,D);
        Si_tcp = zeros(D,D,2);
    else
        pi_tcp = zeros(1,K);
        mu_tcp = zeros(K,D);
        Si_tcp = zeros(D,D,K);
    end
    
    for k = 1:K
        
        % Update parameters
        Qk = sum(q(:,k),1);
        pi_tcp(k) = Qk./M;
        mu_tcp(k,:) = sum(bsxfun(@times, q(:,k), Z),1)./Qk;
        Si_tcp(:,:,k) = (bsxfun(@minus, Z, mu_tcp(k,:))'*diag(q(:,k))*bsxfun(@minus, Z, mu_tcp(k,:)))./Qk;
        
        % Regularization
        if p.Results.lambda>0
            % Constrain covariance to minimum size
            [V,E] = eig((Si_tcp(:,:,k)+Si_tcp(:,:,k)')./2);
            E = max(0, E - p.Results.lambda.*eye(D));
            Si_tcp(:,:,k) = V*E*V' + p.Results.lambda*eye(D);
        end
    end
    
    % Combine class-covariances
    for k = 1:K
        Si_tcp(:,:,k) = Si_tcp(:,:,k)*pi_tcp(k);
    end
    Si_tcp = sum(Si_tcp,3);
    
    %%%% Minimization
    
    % Gradient step
    R_tcp = ll_lda(pi_tcp,mu_tcp,Si_tcp,Z);
    Dq = R_tcp - R_ref;
    
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
    q = q - lr.*Dq;
    
    % Project back onto simplex
    q = projsplx(q);
    
    %%% Check progress
    if rem(n,1e2)==1
        
        % Maximin likelihood
        Rmm_ = mean(sum(R_tcp.*q - R_ref.*q,2),1);
        
        % Inform user
        disp(['Iteration ' num2str(n) '/' num2str(p.Results.maxIter) ' - Minimax likelihood: ' num2str(Rmm)]);
        
        % Check for update under tolerance
        dll = norm(Rmm-Rmm_);
        if  (dll < p.Results.xTol) || isnan(Rmm_)
            disp(['Broke at ' num2str(n)]);
            break;
        end
        
        % Update likelihood
        Rmm = Rmm_;
    end
end

if ~isempty(p.Results.yZ)
    
    % Check for same labels
    yZ = p.Results.yZ;
    if ~iscolumn(yZ); yZ = yZ'; end
    if ~all(unique(yZ)'==labels); error('Different source and target labels'); end
    
    % Preallocate
    if K==1
        pi_orc = zeros(1,2);
        mu_orc = zeros(2,D);
        Si_orc = zeros(D,D,2);
    else
        pi_orc = zeros(1,K);
        mu_orc = zeros(K,D);
        Si_orc = zeros(D,D,K);
    end
    
    for k = 1:K
        
        % Parameters
        Mk = sum(p.Results.yZ==labels(k),1);
        pi_orc(k) = Mk./M;
        mu_orc(k,:) = sum(Z(p.Results.yZ==labels(k),:),1)./Mk;
        Si_orc(:,:,k) = (bsxfun(@minus,Z(p.Results.yZ==labels(k),:),mu_orc(k,:))'*bsxfun(@minus,Z(p.Results.yZ==labels(k),:),mu_orc(k,:)))./Mk;
        
        % Regularization
        if p.Results.lambda>0
            % Constrain covariance to minimum size
            [V,E] = eig((Si_orc(:,:,k)+Si_orc(:,:,k)')./2);
            E = max(0, E - p.Results.lambda.*eye(D));
            Si_orc(:,:,k) = V*E*V' + p.Results.lambda*eye(D);
        end
    end
    
    % Combine class-covariances
    for k = 1:K
        Si_orc(:,:,k) = Si_orc(:,:,k)*pi_orc(k);
    end
    Si_orc = sum(Si_orc,3);
    
    % Output parameter
    theta.orc = {pi_orc,mu_orc,Si_orc};
    
    % Risk for worst-case labeling (average negative log-likelihood)
    R.orc_q = mean(-sum(ll_lda(pi_orc,mu_orc,Si_orc,Z,q),2),1);
    
end

% Output parameters
theta.tcp = {pi_tcp,mu_tcp,Si_tcp};
theta.ref = {pi_ref,mu_ref,Si_ref};

% Risk for found worst-case labeling (average negative log-likelihood)
R.tcp_q = mean(-sum(ll_lda(pi_tcp,mu_tcp,Si_tcp,Z,q),2),1);
R.ref_q = mean(-sum(ll_lda(pi_ref,mu_ref,Si_ref,Z,q),2),1);

if ~isempty(p.Results.yZ)
    
    % Risk for true labeling (average negative log-likelihood)
    R.tcp_u = mean(-sum(ll_lda(pi_tcp,mu_tcp,Si_tcp,Z,p.Results.yZ),2),1);
    R.ref_u = mean(-sum(ll_lda(pi_ref,mu_ref,Si_ref,Z,p.Results.yZ),2),1);
    R.orc_u = mean(-sum(ll_lda(pi_orc,mu_orc,Si_orc,Z,p.Results.yZ),2),1);
    
    % Error on true labeling
    [e.tcp_u, pred.tcp_u, post.tcp_u, AUC.tcp_u] = lda_err(pi_tcp,mu_tcp,Si_tcp,Z,p.Results.yZ);
    [e.ref_u, pred.ref_u, post.ref_u, AUC.ref_u] = lda_err(pi_ref,mu_ref,Si_ref,Z,p.Results.yZ);
    [e.orc_u, pred.orc_u, post.orc_u, AUC.orc_u] = lda_err(pi_orc,mu_orc,Si_orc,Z,p.Results.yZ);
    
    % Output
    varargout{3} = e;
    varargout{4} = pred;
    varargout{5} = post;
    varargout{6} = AUC;
end

% Output
varargout{1} = q;
varargout{2} = R;

end

