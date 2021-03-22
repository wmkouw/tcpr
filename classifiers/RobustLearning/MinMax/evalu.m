% Evaluation function
% Input: K - (t x t) kernel matrix
%        y - (t x 1) labels
%        theta - the model to be evaluated
%        type - character, 'R' for regression, 'C' for classification
%        w - (t x 1) weights for data points (same weights by default)
% Last modified: July 28, 2014
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function [loss, err] = evalu(K, y, theta, type, w)
    t = size(K,1);
    if(nargin < 5)
        w = ones(t,1)./t;
    end
    switch type
        case 'R' % regression
            loss = w'*((K*theta - y).^2);
            err = loss;
        case 'C' % classification
            b = theta(end);
            theta(end) = [];
            yhat = K*theta+b;
            loss = w'*max(0, 1-yhat.*y);
            err = sum(y~=sign(yhat))/t;
    end
end
