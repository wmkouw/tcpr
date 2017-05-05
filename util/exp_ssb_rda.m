function exp_rda(X,yX,Z,yZ,varargin)
% Experiment function to run learning curves for MCPL-LDA

% Parse arguments
p = inputParser;
addOptional(p, 'NN', length(yZ));
addOptional(p, 'R', 1);
addOptional(p, 'lambda', 1);
addOptional(p, 'clip', 100);
addOptional(p, 'iwe', 'kliep');
addOptional(p, 'prep', {''});
addOptional(p, 'maxIter', 500);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'svnm', []);
addOptional(p, 'tst', []);
parse(p, varargin{:});

% Setup for learning curves
NZ = length(yZ);
if ~isempty(p.Results.NN);
    lN = length(p.Results.NN);
else
    lN = 1;
end

% Normalize data to prevent 
X = da_prep(X', p.Results.prep)';
Z = da_prep(Z', p.Results.prep)';

% Preallocate
theta = cell(p.Results.R,lN);
err = zeros(p.Results.R,lN);
pred = cell(p.Results.R,lN);
iw = cell(p.Results.R,lN);
for n = 1:lN
    for r = 1:p.Results.R
        disp(['Running repeat ' num2str(r) '/' num2str(p.Results.R)]);
        
        % Select increasing amount of unlabeled samples
         if ~isempty(p.Results.NN)
            ixU{r,n} = randsample(1:NZ, p.Results.NN(n), false);
        else
            ixU{r,n} = 1:NZ;
         end
        
        % Map labels to {-1,+1}
        yX(yX~=1) = -1;
%         yZ(yZ~=1) = -1;
        
        % Call classifier and evaluate
        [theta{r,n},err(r,n),pred{r,n},iw{r,n}] = rda(X,yX,Z(ixU{r,n},:),'yZ', yZ(ixU{r,n}), 'xTol',p.Results.xTol,'maxIter', p.Results.maxIter, 'lambda', p.Results.lambda, 'clip', p.Results.clip, 'iwe', p.Results.iwe);
        
    end
end

% Write results
di = 1; while exist(['results_rda_' p.Results.svnm num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = ['results_rda_' p.Results.svnm num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'err', 'iw', 'pred','p');

end
