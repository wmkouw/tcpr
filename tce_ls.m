function [theta,varargout] = tce_ls(X,yX,Z,varargin)
% Function to run the Least-Squares version of the Target Contrastive Estimator
% Input:
% 		    X      	source data (N samples x D features)
%           Z      	target data (M samples x D features)
%           yX 	   	source labels (N x 1)
% Optional input:
%     		yZ 		target labels (M samples x 1, for evaluation)
% 			alpha 	learning rate (default: 1)
%           lambda  l2-regularization parameter (default: 0)
% 			maxIter maximum number of iterations (default: 500)
% 			xTol 	convergence criterion (default: 1e-5)
% 			viz		visualization during optimization (default: false)
% Output:   
% 			theta   tce estimate
% Optional output:
%           {1}   	target loss of the tce estimate
% 			{2} 	target loss of the source esimate
% 			{3}		target error of the tce estimate
%			{4}		target predictions of the tce estimate
%			{5}		target error of the source estimate
%			{6}		target predictions of the source estimate
%
% Wouter M. Kouw (2016). Least-Squares Target Contrastive Estimation for Transfer Learning, ICPR.
% Last update: 01-04-2016

% Parse hyperparameters
p = inputParser;    
addOptional(p, 'yZ', []);
addOptional(p, 'alpha', 1);
addOptional(p, 'lambda', 1e-3);
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'viz', 0);
parse(p, varargin{:});

% Augment data with bias if necessary
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end
if ~all(Z(:,end)==1); Z = [Z ones(size(Z,1),1)]; end

% Size
[M,D] = size(Z);
lab = [-1 +1];
K = numel(lab);

% Reference parameter estimates
theta.ref = svdinv(X'*X + p.Results.lambda*eye(D))*X'*yX;

% Initialize
q = ones(M,K)./K;
Dq = zeros(M,K);
theta.mcpl = theta.ref;

disp('Starting MCPL optimization');
ll_old = Inf;
for n = 1:p.Results.maxIter
    
    %%% Minimization
    theta.mcpl = svdinv(Z'*Z + p.Results.lambda*eye(D))*Z'*(-q(:,1)+q(:,2));
    
    %%% Maximization
    % Compute new gradient
    for k = 1:K
        Dq(:,k) = (Z*theta.mcpl - lab(k)).^2 - (Z*theta.ref - lab(k)).^2;
    end
    
    % Apply gradient and project back onto simplex (alpha is accelerating/decelerating constant for 1/n learning rate)
    if n < (p.Results.maxIter./10);
        q = proj_splx((q + Dq.*100)')';
    else
        q = proj_splx((q + Dq./(p.Results.alpha * n))')';
    end
    
    % Visualize
    if p.Results.viz
        if rem(n,500)==2;
            % Scatter first 2 dimensions and decision boundaries
            cm = cool;
            mk = {'x','o'};
            figure(1);
            clf(1)
            hold on
            for j = 1:size(Z,1)
                [~,mky] = max(lab==p.Results.yZ(j),[],2);
                plot(Z(j,1),Z(j,2), 'Color', cm(1+round(q(j,1)*63),:), 'Marker', mk{mky}, 'LineStyle', 'none');
            end
            h_m = plotl(theta.mcpl, 'Color','r', 'LineStyle', '-.');
            h_r = plotl(theta.ref, 'Color','b','LineStyle',':');
            
            legend([h_m h_r], {'MCPL', 'ref'});
            colorbar
            colormap(cool)
            
            drawnow
            pause(.1);
        end
    end
    
    % Minimax loss
    ll = 0;
    for k = 1:K
        ll = ll + sum(q(:,k).*((Z*theta.mcpl - lab(k)).^2)) - sum(q(:,k).*((Z*theta.ref - lab(k)).^2));
    end
    
    % Update or break
    dll = norm(ll_old-ll);
    if  dll < p.Results.xTol; disp(['Broke at ' num2str(n)]); break; end
    ll_old = ll;
    if rem(n,1000)==1; disp(['Iteration ' num2str(n) '/' num2str(p.Results.maxIter) ' - Minimax loss: ' num2str(ll)]); end
    
end

% Evaluate with target labels
if ~isempty(p.Results.yZ);
    
    % Loss
    varargout{1} = sum(double(p.Results.yZ==lab(1)).*(Z*theta.mcpl - lab(1)).^2 + double(p.Results.yZ==lab(2)).*(Z*theta.mcpl - lab(2)).^2);
    varargout{2} = sum(double(p.Results.yZ==lab(1)).*(Z*theta.ref - lab(1)).^2 + double(p.Results.yZ==lab(2)).*(Z*theta.ref - lab(2)).^2);
    
    % Error
    varargout{4} = sign(Z*theta.mcpl);
    varargout{3} = mean(varargout{4}~=p.Results.yZ);
    varargout{6} = sign(Z*theta.ref);
    varargout{5} = mean(varargout{6}~=p.Results.yZ);
    
    % Output worst-case labeling
    varargout{7} = q;
end

end
