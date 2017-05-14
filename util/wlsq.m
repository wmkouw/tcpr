function [theta] = wlsq(X,y,w,varargin)
% Importance weighted least-squares
% Input:
% 		    X      	source data (N samples x D features)
%           y 	   	source labels (N x 1)
%           w       importance weights (N x 1)
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

% Check for augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end

% Shape
[N,D] = size(X);

% Force labels into {-1,+1}
if ~isempty(setdiff(unique(y),[-1 +1]))
    y(y~=1) = -1;
    disp(['Forced labels into {-1,+1}']);
end
        
% Weight samples
bX = bsxfun(@times, w, X);

% Least-squares
theta = (bX'*bX+p.Results.lambda*N*eye(D))\(bX'*y);
        
end
