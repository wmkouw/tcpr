function [err,pred,post,AUC] = qda_err(pi_k, mu_k, Si_k, X,y)
% Function to compute QDA error rate

% Data shape
labels = unique(y)';
K = numel(labels);
[N,~] = size(X);

pk = zeros(N,K);
for k = 1:K
    pk(:,k) = pi_k(k)*max(realmin, mvnpdf(X,mu_k(k,:),Si_k(:,:,k)));
end
post = pk./sum(pk,2,'omitnan');

% Compute mean classification error
[~,pred] = max(post, [], 2);
err = mean(labels(pred)'~=y);

% Compute AUC
if K==2
    [~,~,~,AUC] = perfcurve(y,post(:,2),+1);
else
    AUC = NaN;
    disp('No AUC - K ~=2');    
end

end
