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
addOptional(p, 'gamma', .1);
addOptional(p, 'order', 'first');
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'yZ', []);
addOptional(p, 'clip', 1000);
parse(p, varargin{:});

% Shapes
[N,D] = size(X);
[M,~] = size(Z);

% Check for column vector y
if ~iscolumn(yX); yX = yX'; end

% Labeling
labels = unique(yX)';
K = numel(labels);
if K>2; error('Binary classification only'); end
if ~all(labels==[-1 +1]); error('Labels {-1,+1} expected'); end

% Feature function
switch p.Results.order
    case 'first'
        
        % Feature functions
        fxy = zeros(N,D+1,K);
        fzy = zeros(M,D+1,K);
        for k = 1:K
            fxy(:,:,k) = [labels(k)*X labels(k)*ones(N,1)];
            fzy(:,:,k) = [labels(k)*Z labels(k)*ones(M,1)];
        end
    otherwise
        error('Higher-order moments than first not implemented');
end

% Compute moment-matching constraint
pyx = zeros(N,K);
for i = 1:N
    [~,yi] = max(yX(i)==labels,[],2);
    pyx(i,yi) = 1;
end
c = zeros(D+1,K);
for i = 1:N
    for k = 1:K
        c(:,k) = c(:,k) + pyx(i,k)*fxy(i,:,k)';
    end
end
c = c./N;

% Density ratio
p_src = mvnpdf(X, mean(X,1), cov(X)+p.Results.gamma*eye(D));
p_tgt = mvnpdf(X, mean(Z,1), cov(Z)+p.Results.gamma*eye(D));
iw = p_src./p_tgt;

% Self-normalize importance weights
iw = N*iw./sum(iw);

% Clip weights if necessary
iw = min(p.Results.clip, max(1./p.Results.clip, iw));

% Initialize
theta = randn(D+1,K)*0.001;
for n = 1:p.Results.maxIter
    
    % Calculate weighted feature function
    psi = zeros(N,K);
    for k = 1:K
        for i = 1:N
            psi(i,k) = iw(i).* fxy(i,:,k)*theta(:,k);
        end
    end
    psi(isnan(psi)) = 0;
    
    % Estimate posterior
    pyx = zeros(N,K);
    for k = 1:K
        a = max(psi,[],2);
        pyx(:,k) = exp(psi(:,k)-a)./ sum(exp(bsxfun(@minus,psi,a)),2);
    end
    
    % Estimate expected feature function
    dL = zeros(D+1,K);
    for i = 1:N
        for k = 1:K
            dL = dL + pyx(i,k)*fxy(i,:,k)';
        end
    end
    dL = dL./N;
    
    % Compute gradient
    dC = c - dL;
    
    % Update theta
    update = dC - 2.*theta;
    theta = theta + 1./n.*update;
    
    % Check for step under tolerance
    if norm(update) <= p.Results.xTol
        disp(['Broke at ' num2str(n)]);
        break;
    end
    
    % Report progress
    if rem(n, p.Results.maxIter./10)==1
        fprintf('Iteration %i/%i - Gradient: %.8f\n', n,p.Results.maxIter, norm(update));
    end
    
end

% Evaluate with target labels
if ~isempty(p.Results.yZ)
    
    % Check for column vector
    yZ = p.Results.yZ;
    if ~iscolumn(yZ); yZ = yZ'; end
    if ~all(unique(yZ)'==labels); error('Different source and target labels'); end
    
    % Posteriors for target samples
    post = zeros(M,K);
    for j = 1:M
        parf = 0;
        for k = 1:K
            parf = parf + exp(fzy(j,:,k)*theta(:,k));
        end
        for k = 1:K
            post(j,k) = exp(fzy(j,:,k)*theta(:,k))./parf;
        end
    end
    
    % Risk log-loss
    R = 0;
    for j = 1:M
        [~,yj] = max(yZ(j)==labels,[],2);
        parf = 0;
        for k = 1:K
            parf = parf + exp(fzy(j,:,k)*theta(:,k));
        end
        for k = 1:K
            R = R + -fzy(j,:,yj)*theta(:,yj) + log(parf);
        end
    end
    R = R./M;
    
    % Predictions
    [~,pred] = max(post, [], 2);
    
    % Error
    e = mean(labels(pred)' ~= yZ);
    
    % AUC
    [~,~,~,AUC] = perfcurve(yZ, post(:,K),max(labels));
    
    varargout{1} = R;
    varargout{2} = e;
    varargout{3} = pred;
    varargout{4} = post;
    varargout{5} = AUC;
    
end

end
