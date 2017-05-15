function [theta,iw, varargout] = rba(X,yX,Z,varargin)
% Function that performs Robust Bias-Aware Learning for Domain Adaptation.
%
% Code written per reference:
% Robust Classification under Sample Selection Bias, Anqi Liu & Brian
% Ziebart (2014), NIPS.

% Parse hyperparameters
p = inputParser;
addOptional(p, 'lambda', 1e-3);
addOptional(p, 'sigma', 1);
addOptional(p, 'gamma', 1);
addOptional(p, 'order', 'first');
addOptional(p, 'iwe', 'kmm');
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'yZ', []);
addOptional(p, 'clip', 1000);
parse(p, varargin{:});

% Shapes
[N,D] = size(X);
[M,~] = size(Z);

% Labels
labels = unique(yX)';
K = numel(labels);
if K>2; error('RCSA code only supports binary classification'); end

% Feature function
switch p.Results.order
    case 'first'
        
        % Sufficient statistics
        fxy = zeros(N,D+1,K);
        fzy = zeros(M,D+1,K);
        for k = 1:K
            fxy(yX==labels(k),:,k) = [X(yX==labels(k),:) ones(sum(yX==labels(k)),1)];
            fzy(:,:,k) = [Z ones(M,1)];
        end
        
    case 'second'
    case 'third'
    otherwise
        error('Higher-order moments than third not implemented');
end

% Compute moment-matching constraint
c = squeeze(mean(fxy,1))';

% Calculate importance weights
switch lower(p.Results.iwe)
    case 'kmm'
        iw = iwe_kmm(X,Z, 'theta', p.Results.sigma, 'mD', 'se', ...
            'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, ...
            'gamma', 1);
end

% Clip the inverse weights
iw = max(1./p.Results.clip, min(iw, p.Results.clip));

% Invert p_T(x)/p_S(x) to p_S(x)/p_T(x)
iw = 1./iw;

% Initialize
theta = rand(K,D+1);
for n = 1:p.Results.maxIter
    
    % Calculate posteriors
    psi = zeros(N,K);
    for k = 1:K
        for i = 1:N
            psi(i,k) = iw(i).* theta(k,:) * fxy(i,:,k)';
        end
    end
    psi(isnan(psi)) = 0;
    
    pyx = zeros(N,K);
    dL = zeros(K,D+1);
    for k = 1:K
        a = max(psi,[],2);
        pyx(:,k) = exp(psi(:,k)-a)./ sum(exp(bsxfun(@minus,psi,a)),2);
        for i = 1:N
            dL(k,:) = dL(k,:) + sum(pyx(i,k)'*fxy(i,:,:),3);
        end
    end
    dL = dL./N;
    
    % Compute gradient with moment-matching gradients and regularization
    dC = c - dL - p.Results.lambda.*2.*theta;
    if any(isnan(dC)); error('Numerical explosion'); end
    
    % Update theta
    theta = theta + dC./(n*p.Results.gamma);
    
    % Check for step under tolerance
    if norm(dC) <= p.Results.xTol
        disp(['Broke at ' num2str(n)]);
        break;
    end
    
    % Report progress
    if rem(n,100)==1
        disp(['Iteration ' num2str(n) '/' num2str(p.Results.maxIter) ' - Gradient: ' num2str(norm(dC))]);
    end
    
end

if nargout>1
    
    % Evaluate with target labels
    if ~isempty(p.Results.yZ)
        
        % Error on target set
        post = zeros(M,K);
        for i = 1:M
            for k = 1:K
                post(i,k) = exp(theta(k,:)*fzy(i,:,k)')./sum(exp(sum(theta.*squeeze(fzy(i,:,:))',2)),1);
            end
        end
        
        % Risk is log-loss
        R = 0;
        for j = 1:M
            [~,yi] = max(p.Results.yZ(i)==labels,[],2);
            R = R + -[Z(j,:) 1]*theta(yi,:)' + log(sum(exp([Z(j,:) 1]*theta')));
        end
        R = R./M;
        
        % Predictions
        [~,pred] = max(post, [], 2);
        
        % Error
        e = mean(labels(pred)' ~= p.Results.yZ);
        
        % AUC
        [~,~,~,AUC] = perfcurve(p.Results.yZ, post(:,K),max(labels));
        
        varargout{1} = R;
        varargout{2} = e;
        varargout{3} = pred;
        varargout{4} = post;
        varargout{5} = AUC;
        
    end
end

end
