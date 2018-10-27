function [theta] = wlsq(X,y,iw,varargin)
% Importance weighted least-squares
% Input:
% 		    X      	source data (N samples x D features)
%           y 	   	source labels (N x 1)
%           iw       importance weights (N x 1)
% Optional input:
%           lambda  l2-regularization parameter (default: 0)
% Output:
% 			theta   target model estimate
%
% Wouter Kouw
% Last update: 2017-01-31

% Parse input
p = inputParser;
addOptional(p, 'lambda', 0);
parse(p, varargin{:});

% Check for column vector y
if ~iscolumn(y); y = y'; end

% Labeling
labels = unique(y);
K = numel(labels);
if K>2; error('Binary classification only'); end

% Check for augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end

% Shape
[N, D] = size(X);
        
% Weight samples
bX = bsxfun(@times, iw, X);

% Least-squares
theta = (bX'*bX+p.Results.lambda*eye(D))\(bX'*y);
        
end
