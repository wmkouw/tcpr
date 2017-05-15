function [W,P,varargout] = tca(X,yX,Z,varargin)
% Function to perform Transfer Component Analysis.
% Ref: Pan, Tsang, Kwok, Yang (2009). Domain Adaptation via Transfer Component Analysis.
%
% Input:    X       source data (n samples x D features)
%           yX      source labels (n x 1) in {-1,1}
%           Z       target data (m samples x D features)
% Optional:
%           yZ      target labels
%           l2      l2-regularization parameters (default: 1e-3)
%           nC      Number of components to reduce to (default: 1)
%           mu      trade-off parameter transfer components (default: 1)
%           sigma   radial basis function kernel width (default: 1)
%
% Output:   W       Classifier parameters
%           P       Transfer Components
% Optional:
%           {1}     Risk of classifier
%           {2}     Error of target label predictions
%           {3}     Predictions on target set
%           {4}     Posteriors of target set
%           {5}     AUC of predictions
%
% Copyright: Wouter M. Kouw
% Last update: 04-04-2016

addpath(genpath('~/Dropbox/Libs/minFunc'));

% Shapes
[n,~] = size(X);
[m,~] = size(Z);
labels = unique(yX)';
K = numel(labels);

% Parse input
p = inputParser;
addOptional(p, 'yZ', []);
addOptional(p, 'nC', 1);
addOptional(p, 'l2', 1e-3);
addOptional(p, 'mu', 1./2);
addOptional(p, 'sigma', 1);
addOptional(p, 'maxIter', 100);
addOptional(p, 'xTol', 1e-5);
parse(p, varargin{:});

% Find components
[P,KXZ] = tc(X', Z', 'sigma', p.Results.sigma, 'mu', p.Results.mu, 'nC', p.Results.nC);

% Build parametric kernel map
B = KXZ*P;
clear KXZ

% Train classifier
W = mlr(B(1:n,:), yX, p.Results.l2);

if ~isempty(p.Results.yZ)
    
    % Apply classifier
    ZW = [B(n+1:n+m,:) ones(m,1)]*W;
    
    % Average negative log-likelihood (-ALL)
    R = 0;
    for j = 1:m
        [~,yi] = max(p.Results.yZ(j)==labels,[],2);
        R = R + -ZW(j,yi) + log(sum(exp(ZW(j,:)),2));
    end
    R = R./m;
        
    % Compute predictions
    [~,pred] = max(ZW,[],2);
    
    % Compute error
    e = mean(labels(pred)' ~= p.Results.yZ);
    
    % Compute posteriors
    a = max(ZW,[],2);
    post = exp(bsxfun(@minus, ZW(:,K), a))./sum(exp(bsxfun(@minus, ZW, a)),2);
    
    % Compute AUC
    if K==2
        [~,~,~,AUC] = perfcurve(p.Results.yZ,post,+1);
    else
        AUC = NaN;
        disp('No AUC - K ~=2');    
    end
    
    varargout{1} = R;
    varargout{2} = e;
    varargout{3} = pred;
    varargout{4} = post;
    varargout{5} = AUC;
    
end

end


       



