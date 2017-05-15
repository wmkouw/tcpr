function [ll] = ll_qda(pi_k,mu_k,Si_k,X,q)
% Function to compute log-likelihood of samples under a QDA model

% Data shape
[N,~] = size(X);

% Shapes
if exist('q','var')
    if isvector(q)
        % Classes is number of unique labels
        labels = unique(q)';
        K = numel(labels);
        
        % Change label vector into label matrix
        Q = zeros(N,K);
        for i = 1:N
            [~,yi] = max(q(i)==labels,[],2);
            Q(i,yi) = 1;
        end
        q = Q;
    else
        K = size(q,2);
    end
else
    K = length(pi_k);
end

ll = zeros(N,K);
for k = 1:K
    ll(:,k) = log(pi_k(k))+log(max(realmin, mvnpdf(X,mu_k(k,:),Si_k(:,:,k))));
end

if exist('q','var')
    % Weigh likelihood with soft label
    ll = ll.*q;
end

end

