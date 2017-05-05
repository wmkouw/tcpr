function [ll] = ll_lda(pi_k,mu_k,Si,X,q)
% Function to compute log-likelihood of samples under an LDA model

% Data shape
[N,~] = size(X);

% Shapes
if exist('q','var')
    if isvector(q)
        % Classes is number of unique labels
        K = numel(unique(q));
        
        % Change label vector into label matrix
        Q = zeros(N,K);
        for i = 1:N
            Q(i,q(i)) = 1;
        end
        q = Q;
    else
        K = size(q,2);
    end
else
    K = length(pi_k);
end

% % Eigenvalue decomposition
% [U,S,V] = svd(Si);
% 
% % Inverse of eigenvalues of covariance
% s = diag(S);
% s(s>0) = 1./s(s>0);
% % s = max(realmin,s);
% 
% % Partition function
% C = (-D*log(2*pi)+sum(log(s),1))/2;
% 
% % Log-likelihood
% ll = zeros(N,K);
% for k = 1:K
%     ll(:,k) = C - sum(bsxfun(@minus,X,mu_k(k,:))*(V*diag(sqrt(s))*U').^2,2)/2 + log(pi_k(k));
% end

ll = zeros(N,K);
for k = 1:K
    ll(:,k) = log(pi_k(k))+log(max(realmin,mvnpdf(X,mu_k(k,:),Si)));
end

if exist('q','var')    
    if exist('Q', 'var')
        % Weigh likelihood with hard label
        ll(Q==0) = 0;
    else
        % Weigh likelihood with soft label
        ll = ll.*q;
    end
end

end
