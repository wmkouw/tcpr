function [W] = mlr(X,y,lambda)
% Multiclass logistic regression

addpath(genpath('minFunc'));

% Optimization options
options.Display = 'final';
options.Method = 'lbfgs';
options.DerivativeCheck = 'off';

% Check for bias augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end

% Shapes
[N,D] = size(X);

% Check for column vector y
if ~iscolumn(y); y = y'; end

% Labeling
labels = unique(y)';
K = numel(labels); 

% Minimize loss
w = minFunc(@mlr_grad, zeros(D*K,1), options, X(:,1:end-1),y,lambda);

% Reshape into K-class matrix
W = [reshape(w(1:end-K), [D-1 K]); w(end-K+1:end)'];

end

function [L, dL] = mlr_grad(W,X,y, lambda)
% Multiclass logistic regression gradient

% Shape
[n,D] = size(X);
labels = unique(y);
K = numel(labels); 

% Reshape weight vector
W0 = reshape(W(D*K+1:end), [1 K]);
W = reshape(W(1:D*K), [D K]);

% Compute p(y|x)
XW = bsxfun(@plus, X*W, W0);
XW = exp(bsxfun(@minus, XW, max(XW, [], 2)));
XW = bsxfun(@rdivide, XW, max(sum(XW, 2), realmin));

% Negative log-likelihood of each sample
L = 0;
for i=1:n
    [~,yi] = max(y(i)==labels);
    L = L - log(max(XW(i,yi), realmin));
end
L = L./n + lambda .* sum([W(:); W0(:)] .^ 2);

% Only compute gradient if requested
if nargout > 1
    
    % Compute positive part of gradient
	pos_E = zeros(D, K);
    pos_E0 = zeros(1, K);
    for k=1:K
        pos_E(:,k) = sum(X(y==labels(k),:), 1)';            
        pos_E0(k) = sum(y==labels(k));
    end
    
    % Compute negative part of gradient    
    neg_E = X' * XW;
    neg_E0 = sum(XW, 1)';
        
	% Compute gradient
	dL = -[pos_E(:) - neg_E(:); pos_E0(:) - neg_E0(:)]./n + 2 .* lambda .* [W(:); W0(:)];
    
end
end
