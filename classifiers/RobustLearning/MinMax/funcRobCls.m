function [f, g, alpha, L] = funcRobCls(theta, K, y, Ktr, Kt_t, Kt_ref, Kref_ref, beta, gamma, B)
% Objective function for robust classification with hinge loss
% Input: theta - current model
%        K - kernel matrix for data to be evaluated on
%        y - labels for data to be evaluated on
%        Ktr - (t x t) kernel matrix for training data
%        Kt_t - (t x t) gaussian kernel matrix (training x training)
%        Kt_ref - (t x k) gaussian kernel matrix (training x reference)
%        Kref_ref - (k x k) gaussian kernel matrix (reference x reference)
%        beta - (scalar) parameter for regularization on model theta
%        gamma - (scalar) parameter for soft moment matching constraints
%        B - (scalar) bound on alpha
% Output: f - current objective value
%         g - current gradient
%         alpha - current alpha for inner maximization
%         L - current loss
% Last modified: July 28, 2014
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta

    [~, k] = size(Kt_ref);
    b = theta(end);
    theta(end) = [];
    yhat = K*theta + b;
    l = max(0, 1-yhat.*y); % hinge loss
    f1 = Kt_ref'*l;
    if gamma == 0 % Doug's method (closed form alpha)
        u = sum(Kt_ref, 1)';
        r = f1./u; % f1 is v
        [~, idx] = sort(r);
        u = u(idx);
        j_star = k + 1; u_sum = 0; alpha = zeros(k,1);
        while u_sum < 1/B
            u_sum = u_sum + u(j_star-1);
            j_star = j_star - 1;
        end
        u_sum = u_sum - u(j_star);
        for j = k:-1:(j_star+1)
            alpha(idx(j)) = B;
        end
        alpha(idx(j_star)) = (1-B*u_sum)/u(j_star);
        f = -alpha'*f1;
    else % has soft moment matching constraints
        H = -gamma.*(Kt_ref'*Kt_t*Kt_ref);
        H = (H+H')./2;
        f2 = (gamma/k).*(Kt_ref'*sum(Kt_ref,2));
        Aeq = sum(Kt_ref, 1);
        % old matlab
%         opt = optimset('Algorithm',  'interior-point-convex', 'Display', 'off');
%         [alpha, f, flag] = quadprog(-H,-(f1+f2), [],[],...
%             Aeq,1, zeros(k,1),B.*ones(k,1), [], opt);
        % new matlab (2013a)
%         opt = optimoptions('quadprog', 'Display', 'off');
        [alpha, f, flag] = quadprog(-H,-(f1+f2), [],[],...
            Aeq,1, zeros(k,1),B.*ones(k,1));
    end
    L = f1'*alpha;
    f = -f + beta*(theta'*Ktr*theta)/2 -...
        gamma*sum(sum(Kref_ref))/(2*k*k);
    w = Kt_ref*alpha;
    w(l==0) = 0;
    gYhat = w.*y;
    g = -K'*(gYhat) + beta.*(Ktr*theta);
    g = [g; -sum(gYhat)];
end
