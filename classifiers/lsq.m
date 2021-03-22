function [theta, mse] = ls(X,y,lambda)
% Least-squares classifier with mean squared error

% Check for regularization
if ~exist('lambda','var'); lambda = 0; end

% Check for augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end

% Coefficients
theta = (X'*X + lambda.*eye(size(X,2)))\(X'*y);

% Mean Squared Error
mse = mean((y - X*theta).^2);

end
