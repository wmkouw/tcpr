% Objective function for regular classification with hinge loss
% Input: theta - current model
%        K - kernel matrix for data to be evaluated on
%        y - labels for data to be evaluated on
%        beta - (scalar) parameter for regularization on model theta
%        w - weights for data points
% Output: f - current objective value
%         g - current gradient
%         L - current loss
% Last modified: July 28, 2014
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function [f, g, L] = funcNonRobCls(theta, K, y, beta, w)
    t = size(K, 1);
    b = theta(end);
    theta(end) = [];
    if(nargin < 5)
        w = ones(t,1)./t;
    end
    yhat = K*theta + b;
    l = max(0, 1-yhat.*y);
    L = sum(l.*w);
    f = L + beta*(theta'*(K)*theta)/2;
    idx = (l~=0);
    gYhat = idx.*y.*w;
    g = -K*gYhat + beta.*((K)*theta);
    g = [g; -sum(gYhat)];
end
