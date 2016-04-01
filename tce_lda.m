function [theta,varargout] = tce_lda(X,yX,Z,varargin)
% Function to run the Linear Discriminant Analysis version of the Target Contrastive Estimator
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
%           {1}   	target likelihood of the tce estimate
% 			{2} 	target likelihood of the source esimate
% 			{3}		target error of the tce estimate
%			{4}		target predictions of the tce estimate
%			{5}		target error of the source estimate
%			{6}		target predictions of the source estimate
%
% Wouter M. Kouw (2016). Target Contrastive Estimator for Robust Domain Adaptation, UAI. 
% Last update: 01-04-2016

% Parse hyperparameters
p = inputParser;
addOptional(p, 'yZ', []);
addOptional(p, 'alpha', 1);
addOptional(p, 'lambda', 0);
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'viz', false);
parse(p, varargin{:});

% Sizes
[N,D] = size(X);
M = size(Z,1);
uy = unique(yX);
K = numel(uy);

% Reference parameter estimates
Nk = zeros(1,K);
pi_ref = NaN(1,K);
mu_ref = NaN(K,D);
S_ref = zeros(D);
for k = 1:K
    Nk(k) = sum(yX==uy(k));
    pi_ref(k) = Nk(k)./N;
    mu_ref(k,:) = mean(X(yX==uy(k),:),1);
    S_ref = S_ref + (bsxfun(@minus,X(yX==uy(k),:), mu_ref(k,:))'*bsxfun(@minus,X(yX==uy(k),:), mu_ref(k,:)))./N;
end
La_ref = svdinv((S_ref+S_ref')./2);

% Precompute log-likelihood of unlabeled samples under reference model
ll_ref = ll_lda(pi_ref,mu_ref,La_ref,Z);

% Initialize target posterior
q = min(max(myprojsplx(ll_ref), realmin), 1-realmin);

disp('Starting MCPL optimization');
llmm = Inf;
for n = 1:p.Results.maxIter
    
    %%% Maximization
    pi_mcpl = NaN(1,K);
    mu_mcpl = NaN(K,D);
    S_mcpl = zeros(D,D,K);
    for k = 1:K;
        pi_mcpl(k) = sum(q(:,k),1)./M;
        mu_mcpl(k,:) = sum(bsxfun(@times, q(:,k), Z),1)./sum(q(:,k),1);
        S_mcpl(:,:,k) = sum(q(:,k),1).*mu_mcpl(k,:)'*mu_mcpl(k,:);
    end
    S_mcpl = (bsxfun(@times,sum(q,2),Z)'*Z - sum(S_mcpl,3))./M;
    
    % Perform singular value decomposition of covariance matrix
    [U_mcpl,S_mcpl,V_mcpl] = svd((S_mcpl+S_mcpl')./2);
    
    % Stable inverse
    S_mcpl(S_mcpl>0) = 1./S_mcpl(S_mcpl>0);
    
    %%%% Minimization
    
    % Compute new gradient
    ll_mcpl = ll_lda(pi_mcpl,mu_mcpl,{U_mcpl,S_mcpl,V_mcpl},Z);
    Dq = ll_mcpl - ll_ref;
    
    % Apply gradient and project back onto simplex
    q = min(max(myprojsplx(q - Dq./(p.Results.alpha+n)), realmin), 1-realmin);
    
    % Visualize
    if p.Results.viz
        if rem(n,100)==1;
            cm = cool;
            mk = {'x','o'};
            figure(1);
            clf(1)
            hold on
            for j = 1:size(Z,1)
                plot(Z(j,1),Z(j,2), 'Color', cm(1+round(q(j,1)*63),:), 'Marker', mk{p.Results.yZ(j)}, 'LineStyle', 'none');
            end
            drawnow
            pause(.1);
        end
    end
    
    % Break or update
    llmm_ = ll_mcpl.*q;
    
    dll = norm(llmm-llmm_);
    if isnan(dll); error('Numeric error'); end
    if  dll < p.Results.xTol; disp(['Broke at ' num2str(n)]); break; end
    llmm = llmm_;
    if rem(n,50)==1; disp(['Iteration ' num2str(n) '/' num2str(p.Results.maxIter) ' - Minimax gradient: ' num2str(dll)]); end
    
end

% Output parameters
La_mcpl = (V_mcpl*S_mcpl*U_mcpl');
theta.mcpl = {pi_mcpl,mu_mcpl,La_mcpl};
theta.ref = {pi_ref,mu_ref,La_ref};

% Evaluate with target labels
if ~isempty(p.Results.yZ);
    
    % Loss
    varargout{1} = sum(sum(ll_lda(pi_mcpl,mu_mcpl,La_mcpl,Z,p.Results.yZ),2),1);
    varargout{2} = sum(sum(ll_lda(pi_ref,mu_ref,La_ref,Z,p.Results.yZ),2),1);
    
    % Error
    [varargout{3},varargout{4}] = lda_err(Z,p.Results.yZ,theta.mcpl{2}, theta.mcpl{3});
    [varargout{5},varargout{6}] = lda_err(Z,p.Results.yZ,theta.ref{2}, theta.ref{3});
    
    varargout{7} = q;
end

end
