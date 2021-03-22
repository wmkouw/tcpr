% Objective function for regular regression with L_2 loss
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
function [f, g, L] = funcNonRobReg(theta, K, y, beta, w)
    t = size(K, 1);
    if(nargin < 5)
        w = ones(t,1)./t;
    end
    yhat = K*theta;
    l = yhat - y;
    L = sum((l.^2).*w);
    f = L + beta*(theta'*K*theta)/2;
    g = 2.*(K*(l.*w)) + beta.*(K*theta);
end
