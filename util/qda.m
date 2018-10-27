function [theta,l] = qda(X,y,lambda)
% Function to estimate parameters for Quadratic Discriminant Analysis

if ~exist('lambda','var'); lambda = 0; end

% Shape
[N,D] = size(X);
uy = unique(y);
K = numel(uy);

if K==1
    % Initialize
    pi_k = zeros(2,1);
    mu_k = zeros(2,D);
    Si_k = zeros(D,D,2);
else
    % Initialize
    pi_k = zeros(K,1);
    mu_k = zeros(K,D);
    Si_k = zeros(D,D,K);
end

for k = 1:K
    
    % Priors
    pi_k(k) = sum(y==uy(k))./N;
    
    % Means
    mu_k(k,:) = mean(X(y==uy(k),:),1);
    
    % Covariances
    Si_k(:,:,k) = cov(X(y==uy(k),:));
    
    % Constrain covariance to minimum size
    [V,E] = eig((Si_k(:,:,k)+Si_k(:,:,k)')./2);
    E = max(0, E - lambda.*eye(D));
    Si_k(:,:,k) = V*E*V';
    
    % Regularization
    Si_k(:,:,k) = Si_k(:,:,k) + lambda*eye(D);
end

% Output
theta = {pi_k,mu_k,Si_k};

if nargin > 1
    
    % Compute sample average log-likelihood
    l = sum(sum(ll_qda(pi_k,mu_k,Si_k,X,y),2),1)./N;
end

end
