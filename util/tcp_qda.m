function [theta,varargout] = tcp_qda(X,yX,Z,varargin)
% Quadratic Discriminant Analysis version of the Target Contrastive Pessimistic Estimator
% Input:
% 		    X      	source data (N samples x D features)
%           Z      	target data (M samples x D features)
%           yX 	   	source labels (N x 1)
% Optional input:
%     		yZ 		target labels (M samples x 1, for evaluation)
% 			alpha 	learning rate (default: 1)
%           lambda  l2-regularization parameter (default: 0)
% 			maxIter maximum number of iterations (default: 500)
% 			xTol 	convergence criterion (default: 1e-5)
% 			viz		visualization during optimization (default: false)
% Output:
% 			theta   target model estimate
% Optional output:
%           {1}     found worst-case labeling q
%           {2}   	target loss of the mcpl/ref estimate with q/u
% 			{3} 	target err/pred of the mcpl/ref esimate with u
%
% Wouter M. Kouw (2017). Target Contrastive Pessmistic Risk Estimator.
% Last update: 25-01-2017

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
uy = unique(yX);
K = numel(uy);

% Preallocation
if K==1
    pi_ref = zeros(1,2);
    mu_ref = zeros(2,D);
    Si_ref = repmat(eye(D,D), [1 1 2]);
else
    pi_ref = zeros(1,K);
    mu_ref = zeros(K,D);
    Si_ref = repmat(eye(D,D), [1 1 K]);
end

for k = 1:K
    
    % Parameters
    Nk = sum(yX==uy(k));
    pi_ref(k) = Nk./N;
    mu_ref(k,:) = sum(X(yX==uy(k),:),1)./Nk;
    Si_ref(:,:,k) = (bsxfun(@minus,X(yX==uy(k),:),mu_ref(k,:))'*bsxfun(@minus,X(yX==uy(k),:),mu_ref(k,:)))./Nk;
    
    % Regularization
    if p.Results.lambda>0
        % Constrain covariance to minimum size
        [V,E] = eig((Si_ref(:,:,k)+Si_ref(:,:,k)')./2);
        E = max(0, E - p.Results.lambda.*eye(D));
        Si_ref(:,:,k) = V*E*V' + p.Results.lambda*eye(D);
    end
end

% Log-likelihood of unlabeled samples under reference model
ll_ref = ll_qda(pi_ref,mu_ref,Si_ref,Z);

% Initialize target posterior
q = ones(M,K)./K;

% Initialize mcpl estimates
llmm = Inf;
disp('Starting MCPL optimization');
for n = 1:p.Results.maxIter
    
    %%% Maximization
    if K==1
        pi_tcp = zeros(1,2);
        mu_tcp = zeros(2,D);
        Si_tcp = repmat(eye(D,D), [1 1 2]);
    else
        pi_tcp = zeros(1,K);
        mu_tcp = zeros(K,D);
        Si_tcp = repmat(eye(D,D), [1 1 K]);
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
    
    %%%% Minimization
    
    % Gradient step
    ll_tcp = ll_qda(pi_tcp,mu_tcp,Si_tcp,Z);
    Dq = ll_tcp - ll_ref;
    
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
        llmm_ = sum(sum(ll_tcp.*q - ll_ref.*q,2),1)./M;
        
        % Inform user
        disp(['Iteration ' num2str(n) '/' num2str(p.Results.maxIter) ' - Minimax likelihood: ' num2str(llmm)]);
        
        % Check for update under tolerance
        dll = norm(llmm-llmm_);
        if  (dll < p.Results.xTol) || isnan(llmm_)
            disp(['Broke at ' num2str(n)]);
            break;
        end
        
        % Update likelihood
        llmm = llmm_;
    end
    
end

% Oracle parameter estimates
if ~isempty(p.Results.yZ)
    
    % Unique target labels
    uyZ = unique(p.Results.yZ);
    K = numel(uyZ);
    
    % Preallocate
    if K==1
        pi_orc = zeros(1,2);
        mu_orc = zeros(2,D);
        Si_orc = repmat(eye(D,D), [1 1 2]);
    else
        pi_orc = zeros(1,K);
        mu_orc = zeros(K,D);
        Si_orc = repmat(eye(D,D), [1 1 K]);
    end
    
    for k = 1:K
        
        % Parameters
        Mk = sum(p.Results.yZ==uyZ(k),1);
        pi_orc(k) = Mk./M;
        mu_orc(k,:) = sum(Z(p.Results.yZ==uyZ(k),:),1)./Mk;
        Si_orc(:,:,k) = (bsxfun(@minus,Z(p.Results.yZ==uyZ(k),:),mu_orc(k,:))'*bsxfun(@minus,Z(p.Results.yZ==uyZ(k),:),mu_orc(k,:)))./Mk;
        
        % Regularization
        if p.Results.lambda>0
            % Constrain covariance to minimum size
            [V,E] = eig((Si_orc(:,:,k)+Si_orc(:,:,k)')./2);
            E = max(0, E - p.Results.lambda.*eye(D));
            Si_orc(:,:,k) = V*E*V' + p.Results.lambda*eye(D);
        end
    end
end

% Output parameters
theta.tcp = {pi_tcp,mu_tcp,Si_tcp};
theta.ref = {pi_ref,mu_ref,Si_ref};
theta.orc = {pi_orc,mu_orc,Si_orc};

%%% Optional output
if nargout > 1
    
    % Loss on found worst-case labeling
    ll.tcp_q = mean(sum(ll_qda(pi_tcp,mu_tcp,Si_tcp,Z,q),2),1);
    ll.ref_q = mean(sum(ll_qda(pi_ref,mu_ref,Si_ref,Z,q),2),1);
    ll.orc_q = mean(sum(ll_qda(pi_orc,mu_orc,Si_orc,Z,q),2),1);
    
    if ~isempty(p.Results.yZ)
        
        % Loss on true labeling
        ll.tcp_u = mean(sum(ll_qda(pi_tcp,mu_tcp,Si_tcp,Z,p.Results.yZ),2),1);
        ll.ref_u = mean(sum(ll_qda(pi_ref,mu_ref,Si_ref,Z,p.Results.yZ),2),1);
        ll.orc_u = mean(sum(ll_qda(pi_orc,mu_orc,Si_orc,Z,p.Results.yZ),2),1);
        
        % Error on true labeling
        [e.tcp_u, pred.tcp_u, post.tcp_u, AUC.tcp_u] = qda_err(pi_tcp,mu_tcp,Si_tcp,Z,p.Results.yZ);
        [e.ref_u, pred.ref_u, post.ref_u, AUC.ref_u] = qda_err(pi_ref,mu_ref,Si_ref,Z,p.Results.yZ);
        [e.orc_u, pred.orc_u, post.orc_u, AUC.orc_u] = qda_err(pi_orc,mu_orc,Si_orc,Z,p.Results.yZ);
        
        % Output
        varargout{3} = e;
        varargout{4} = pred;
        varargout{5} = post;
        varargout{6} = AUC;
    end
    
    % Output
    varargout{1} = q;
    varargout{2} = ll;
end

end

