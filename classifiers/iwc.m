function [theta,iw,varargout] = iwc(X,yX,Z,varargin)
% Function to run an importance weighted classifier
% Input:    X       source data (N samples x D features)
%           Z       target data (M samples x D features)
%           yX      source labels (N x 1)
% Optional:
%           yZ      target labels (M x 1)
%           lambda  regularization parameters
%           iwe     importance weight estimation method
%           sigma   kernel bandwidth for importance-weight estimator
%           gamma   regularization for importance-weight estimator
%           clf     loss function
%           clip    maximum importance weight value
%
% Output:
% 			theta   target model parameter estimate
%           iw      importance-weights
% Optional output:
%           {1}   	target risk
% 			{2} 	target error
%           {3}     target predictions
%           {4}     target posteriors
%           {5}     target AUC
%
% Wouter M. Kouw
% Last update: 2017-01-31

% Parse optionals
p = inputParser;
addOptional(p, 'yZ', []);
addOptional(p, 'lambda', 0);
addOptional(p, 'iwe', 'kmm');
addOptional(p, 'gamma', 0);
addOptional(p, 'sigma', 1);
addOptional(p, 'clf', 'lsq');
addOptional(p, 'maxIter', 1e3);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'clip', 1e5);
parse(p, varargin{:});

% Check for column vector y
if ~iscolumn(yX); yX = yX'; end

% Labeling
labels = unique(yX)';
K = numel(labels);
if K>2; error('Binary classification only'); end
if ~all(labels==[-1 +1]); error('Labels {-1,+1} expected'); end

% Augment data with bias if necessary
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end
if ~all(Z(:,end)==1); Z = [Z ones(size(Z,1),1)]; end

% Estimate importance weights
switch p.Results.iwe
    case 'kmm'
        iw = iwe_kmm(X(:,1:end-1),Z(:,1:end-1), ...
            'theta', p.Results.sigma, 'mD', 'se', 'gamma', p.Results.gamma, ...
            'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol);
end

% Clip the inverse weights
iw = max(1./p.Results.clip, min(iw, p.Results.clip));

switch p.Results.clf
    case 'lsq'
        % Weighted least-squares
        theta = wlsq(X,yX,iw,p.Results.lambda);
    case 'lr'
        % Minimize loss
        theta = wlr(X,yX,iw,p.Results.lambda);
    otherwise
        error('Reweighted loss function not implemented');
end

if ~isempty(p.Results.yZ)
    
    % Check for same labels
    yZ = p.Results.yZ;
    if ~iscolumn(yZ); yZ = yZ'; end
    if ~all(unique(yZ)'==labels); error('Different source and target labels'); end
    
    % Compute risk
    switch p.Results.clf
        case 'lsq'
            
            % Risk = Mean Squared Error (MSE)
            R = mean((Z*theta - yZ).^2,1);
            
            % Posteriors
            post = exp(Z*theta)./(exp(-Z*theta) + exp(Z*theta));
            
            % Predictions
            pred = sign(Z*theta);
            
            % Errors
            e = mean(pred~=yZ);
            
            % Compute AUC
            if K==2
                [~,~,~,AUC] = perfcurve(yZ,post,+1);
            else
                AUC = NaN;
                disp('No AUC - K ~=2');
            end
            
        case 'lr'
            
            % Risk = average negative log-likelihood (-ALL)
            R = mean(-yZ.*(Z*theta) + log(exp(-Z*theta) + exp(Z*theta)),1);
            
            % Posteriors
            post = exp(Z*theta)./(exp(-Z*theta) + exp(Z*theta));
            
            % Predictions
            pred = sign(Z*theta);
            
            % Errors
            e = mean(pred ~= yZ);
            
            % Compute AUC
            if K==2
                [~,~,~,AUC] = perfcurve(yZ,post,+1);
            else
                AUC = NaN;
                disp('No AUC - K ~=2');
            end
            
    end
    
    % Output
    varargout{1} = R;
    varargout{2} = e;
    varargout{3} = pred;
    varargout{4} = post;
    varargout{5} = AUC;
end


end

