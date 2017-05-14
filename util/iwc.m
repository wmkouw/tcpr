function [theta,varargout] = iwc(X,yX,Z,varargin)
% Function to run an importance weighted classifier
% Input:    X       source data (N samples x D features)
%           Z       target data (M samples x D features)
%           yX      source labels (N x 1)
% Optional:
%           yZ      target labels (M x 1)
%           lambda  Regularization parameters (default: 1e-3)
%           iwe     Choice of importance weight estimation method (default: 'log')
%           clf     Choice of loss function (default: 'lsq')
%
% Output:
% 			theta   target model estimate
% Optional output:
%           {1}     found importance weights w
%           {2}   	target risk R(Z,yZ)
% 			{3} 	target error
%           {4}     target predictions
%           {5}     target posteriors
%           {6}     target AUC
%
% Wouter M. Kouw
% Last update: 2017-01-31

% Parse optionals
p = inputParser;
addOptional(p, 'yZ', []);
addOptional(p, 'lambda', 0);
addOptional(p, 'sigma', 1);
addOptional(p, 'iwe', 'kmm');
addOptional(p, 'clf', 'lsq');
addOptional(p, 'maxIter', 1e3);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'clip', 1e5);
parse(p, varargin{:});

% Augment data with bias if necessary
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end
if ~all(Z(:,end)==1); Z = [Z ones(size(Z,1),1)]; end

% Data shape
K = numel(unique(yX));
if K>2; error('Binary classification only'); end

% Estimate importance weights
switch p.Results.iwe
    case 'kmm'
        iw = iwe_kmm(X,Z, 'theta', p.Results.sigma, 'mD', 'se', ...
            'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, ...
            'gamma', 1e-3);
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

%%% Optional output
if nargout > 1
    
    if ~isempty(p.Results.yZ)
        yZ = p.Results.yZ;
        
        % Compute risk
        switch p.Results.clf
            case 'lsq'
                % Set labels
                yZ(yZ~=1) = -1;
                
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
                % Set labels
                yZ(yZ~=1) = 0;
                
                % Risk = average negative log-likelihood (-ALL)
                R = mean(-yZ.*(Z*theta) + log(1 + exp(Z*theta)),1);
                
                % Posteriors
                post = exp(Z*theta)./(1 + exp(Z*theta));
                
                % Predictions
                pred = sign(Z*theta); 
                
                % Errors
                pred(pred==-1) = 0;
                e = mean(pred~=yZ);
                
                % Compute AUC
                if K==2
                    [~,~,~,AUC] = perfcurve(yZ,post,+1);
                else
                    AUC = NaN;
                    disp('No AUC - K ~=2');    
                end
                
        end
        
        % Output
        varargout{2} = R;
        varargout{3} = e;
        varargout{4} = pred;
        varargout{5} = post;
        varargout{6} = AUC;
    end
    
    % Output
    varargout{1} = iw;
end


end

