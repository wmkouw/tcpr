function exp_ssb_tcp(D,y, varargin)
% Sample selection bias experiment for TCP

% Size of dataset
[M,~] = size(D);

% Parse arguments
p = inputParser;
addOptional(p, 'clf', 'tcp-lda');
addOptional(p, 'nN', min(10, round(M./2)));
addOptional(p, 'nM', M);
addOptional(p, 'setDiff', false);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 5);
addOptional(p, 'lr', 'geom');
addOptional(p, 'maxIter', 1e3);
addOptional(p, 'xTol', 1e-5);
addOptional(p, 'prep', {''});
addOptional(p, 'lambda', 0);
addOptional(p, 'alpha', 2);
addOptional(p, 'saveName', []);
addOptional(p, 'ssb', 'sdw');
addOptional(p, 'viz', false);
parse(p, varargin{:});

% Normalize data
D = da_prep(D, p.Results.prep);

% Check for column vector y
if ~iscolumn(y); y = y'; end

% Labeling
labels = unique(y)';
K = numel(labels);

% Number of sample sizes
lNN = length(p.Results.nN);
if isempty(p.Results.nM)
    lNM = 1;
else
    lNM = length(p.Results.nM);
end

% Preallocation
theta = cell(p.Results.nR,lNN,lNM);
e = cell(p.Results.nR,lNN,lNM);
R = cell(p.Results.nR,lNN,lNM);
AUC = cell(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
q = cell(p.Results.nR,lNN,lNM);

for r = 1:p.Results.nR
    disp(['Running repeat ' num2str(r) '/' num2str(p.Results.nR)]);
    
    for n = 1:lNN
        
        % Select samples
        switch p.Results.ssb
            case 'nn'
                ix = ssb_nn(D,y,p.Results.nN(n), 'viz', p.Results.viz);
            case 'sdw'
                ix = ssb_sdw(D,y,p.Results.nN(n), 'viz', p.Results.viz);
            otherwise
                error(['Selection bias type ' p.Results.ssb ' unknown']);
        end
        
        % Select source
        X = D(ix,:);
        yX = y(ix);

        % Select target
        if p.Results.setDiff
            Z = D(setdiff(1:M,ix),:);
            yZ = y(setdiff(1:M,ix));
        else
            Z = D;
            yZ = y;
        end
        
        for m = 1:lNM
            
            % Select target samples
            if ~isempty(p.Results.nM)
                [~,ixnM] = datasample(Z, p.Results.nM(m), 'Replace', false);
            else
                ixnM = 1:size(Z,1);
            end
            
            % Cross-validate regularization parameter
            if isempty(p.Results.lambda)
                disp(['Cross-validating for regularization parameter']);
                
                % Set range of regularization parameter
                Lambda = [10.^[-6:1:3]];
                R_la = zeros(1,length(Lambda));
                for la = 1:length(Lambda)
                    
                    % Split folds
                    ixf = randsample(1:p.Results.nF, p.Results.nN(n), true);
                    for f = 1:p.Results.nF
                        
                        switch p.Results.clf
                            case 'tcp-ls'
                                % Train on included folds
                                theta_f = tcp_ls(X(ixf~=f,:),yX(ixf~=f),Z(ixnM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', Lambda(la), 'lr', p.Results.lr);
                                
                                % Evaluate on held-out source folds (MSE)
                                R_la(la) = mean(([X(ixf==f,:) ones(sum(ixf==f),1)]*theta_f.tcp - yX(ixf==f)).^2);
                            case 'tcp-lda'
                                % Train on included folds
                                theta_f = tcp_lda(X(ixf~=f,:),yX(ixf~=f),Z(ixnM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', Lambda(la), 'lr', p.Results.lr);
                                
                                % Evaluate on held-out source folds (-ALL)
                                R_la(la) = R_la(la) - sum(sum(ll_lda(theta_f.tcp{1}, theta_f.tcp{2}, theta_f.tcp{3}, X(ixf==f,:), yX(ixf==f)),2),1)./sum(ixf==f);
                            case 'tcp-qda'
                                % Train on included folds
                                theta_f = tcp_qda(X(ixf~=f,:),yX(ixf~=f),Z(ixnM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', Lambda(la), 'lr', p.Results.lr);
                                
                                % Evaluate on held-out source folds (-ALL)
                                R_la(la) = R_la(la) - sum(sum(ll_qda(theta_f.tcp{1}, theta_f.tcp{2}, theta_f.tcp{3}, X(ixf==f,:), yX(ixf==f)),2),1)./sum(ixf==f);
                        end
                    end
                end
                % Select minimal
                R_la(isinf(R_la)) = NaN;
                [~,ixMinLambda] = min(R_la);
                lambda = Lambda(ixMinLambda);
            else
                lambda = p.Results.lambda;
            end
            disp(['\lambda = ' num2str(lambda)]);
            
            % Call classifier and evaluate
            switch p.Results.clf
                case 'tcp-ls'
                    % Train on source set and test on target set
                    [theta{r,n,m},q{r,n,m},R{r,n,m},e{r,n,m},pred{r,n,m},post{r,n,m},AUC{r,n,m}] = tcp_ls(X,yX,Z(ixnM,:), 'yZ', yZ(ixnM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', lambda, 'lr', p.Results.lr);
                    
                    Dt_t = [D ones(M,1)]*theta{r,n,m}.tcp;
                    Dt_r = [D ones(M,1)]*theta{r,n,m}.ref;
                    Dt_o = [D ones(M,1)]*theta{r,n,m}.orc;
                    
                    % Measure on full set (mean squared error)
                    R{r,n,m}.tcp_a = mean((Dt_t - y).^2,1);
                    R{r,n,m}.ref_a = mean((Dt_r - y).^2,1);
                    R{r,n,m}.orc_a = mean((Dt_o - y).^2,1);
                    
                    % Posteriors
                    post{r,n,m}.tcp_a = exp(Dt_t)./(exp(-Dt_t) + exp(Dt_t));
                    post{r,n,m}.ref_a = exp(Dt_r)./(exp(-Dt_r) + exp(Dt_r));
                    post{r,n,m}.orc_a = exp(Dt_o)./(exp(-Dt_o) + exp(Dt_o));
                    
                    % Predictions
                    pred{r,n,m}.tcp_a = sign(Dt_t);
                    pred{r,n,m}.ref_a = sign(Dt_r);
                    pred{r,n,m}.orc_a = sign(Dt_o);
                    
                    % Error on true labeling
                    e{r,n,m}.tcp_a = mean(pred{r,n,m}.tcp_a ~= y);
                    e{r,n,m}.ref_a = mean(pred{r,n,m}.ref_a ~= y);
                    e{r,n,m}.orc_a = mean(pred{r,n,m}.orc_a ~= y);
                    
                    % AUC on true labeling
                    [~,~,~,AUC{r,n,m}.tcp_a] = perfcurve(y,post{r,n,m}.tcp_a,+1);
                    [~,~,~,AUC{r,n,m}.ref_a] = perfcurve(y,post{r,n,m}.ref_a,+1);
                    [~,~,~,AUC{r,n,m}.orc_a] = perfcurve(y,post{r,n,m}.orc_a,+1);
                    
                case 'tcp-lda'
                    % Train on source set and test on target set
                    [theta{r,n,m},q{r,n,m},R{r,n,m},e{r,n,m},pred{r,n,m},post{r,n,m},AUC{r,n,m}] = tcp_lda(X,yX,Z(ixnM,:), 'yZ', yZ(ixnM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', lambda, 'lr', p.Results.lr);
                    
                    % Measure on full set (average negative log-likelihood)
                    R{r,n,m}.tcp_a = mean(-sum(ll_lda(theta{r,n,m}.tcp{1},theta{r,n,m}.tcp{2},theta{r,n,m}.tcp{3},D,y),2),1);
                    R{r,n,m}.ref_a = mean(-sum(ll_lda(theta{r,n,m}.ref{1},theta{r,n,m}.ref{2},theta{r,n,m}.ref{3},D,y),2),1);
                    R{r,n,m}.orc_a = mean(-sum(ll_lda(theta{r,n,m}.orc{1},theta{r,n,m}.orc{2},theta{r,n,m}.orc{3},D,y),2),1);
                    [e{r,n,m}.tcp_a, pred{r,n,m}.tcp_a, post{r,n,m}.tcp_a, AUC{r,n,m}.tcp_a] = lda_err(theta{r,n,m}.tcp{1},theta{r,n,m}.tcp{2},theta{r,n,m}.tcp{3},D,y);
                    [e{r,n,m}.ref_a, pred{r,n,m}.ref_a, post{r,n,m}.ref_a, AUC{r,n,m}.ref_a] = lda_err(theta{r,n,m}.ref{1},theta{r,n,m}.ref{2},theta{r,n,m}.ref{3},D,y);
                    [e{r,n,m}.orc_a, pred{r,n,m}.orc_a, post{r,n,m}.orc_a, AUC{r,n,m}.orc_a] = lda_err(theta{r,n,m}.orc{1},theta{r,n,m}.orc{2},theta{r,n,m}.orc{3},D,y);
                case 'tcp-qda'
                    % Train on source set and test on target set
                    [theta{r,n,m},q{r,n,m},R{r,n,m},e{r,n,m},pred{r,n,m},post{r,n,m},AUC{r,n,m}] = tcp_qda(X,yX,Z(ixnM,:), 'yZ', yZ(ixnM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', lambda, 'lr', p.Results.lr);
                    
                    % Measure on full set (average negative log-likelihood)
                    R{r,n,m}.tcp_a = mean(-sum(ll_qda(theta{r,n,m}.tcp{1},theta{r,n,m}.tcp{2},theta{r,n,m}.tcp{3},D,y),2),1);
                    R{r,n,m}.ref_a = mean(-sum(ll_qda(theta{r,n,m}.ref{1},theta{r,n,m}.ref{2},theta{r,n,m}.ref{3},D,y),2),1);
                    R{r,n,m}.orc_a = mean(-sum(ll_qda(theta{r,n,m}.orc{1},theta{r,n,m}.orc{2},theta{r,n,m}.orc{3},D,y),2),1);
                    [e{r,n,m}.tcp_a, pred{r,n,m}.tcp_a, post{r,n,m}.tcp_a, AUC{r,n,m}.tcp_a] = qda_err(theta{r,n,m}.tcp{1},theta{r,n,m}.tcp{2},theta{r,n,m}.tcp{3},D,y);
                    [e{r,n,m}.ref_a, pred{r,n,m}.ref_a, post{r,n,m}.ref_a, AUC{r,n,m}.ref_a] = qda_err(theta{r,n,m}.ref{1},theta{r,n,m}.ref{2},theta{r,n,m}.ref{3},D,y);
                    [e{r,n,m}.orc_a, pred{r,n,m}.orc_a, post{r,n,m}.orc_a, AUC{r,n,m}.orc_a] = qda_err(theta{r,n,m}.orc{1},theta{r,n,m}.orc{2},theta{r,n,m}.orc{3},D,y);
            end
        end
    end
end

% Write results
di = 1; while exist([p.Results.saveName 'results_ssb_' p.Results.clf '_' num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = [p.Results.saveName 'results_ssb_' p.Results.clf '_' num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'q','R', 'e', 'post', 'AUC', 'lambda', 'p');

end
