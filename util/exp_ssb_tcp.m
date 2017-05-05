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
addOptional(p, 'svnm', []);
addOptional(p, 'ssb', 'nn');
addOptional(p, 'viz', false);
parse(p, varargin{:});

% Normalize data
D = da_prep(D, p.Results.prep);

if strcmp(p.Results.clf, 'tcp-ls')
    % Force labels in {-1,+1}
    lab = unique(y);
    if ~isempty(setdiff(lab,[-1 1]))
        disp(['Forcing labels into {-1,+1}']);
        y(y~=1) = -1;
    end
    lab = unique(y);
end

% Number of sample sizes
lNN = length(p.Results.nN);
if isempty(p.Results.nM)
    lNM = 1;
else
    lNM = length(p.Results.nM);
end

% Preallocation
theta = cell(p.Results.nR,lNN,lNM);
q = cell(p.Results.nR,lNN,lNM);
e = cell(p.Results.nR,lNN,lNM);
R = cell(p.Results.nR,lNN,lNM);
pred = cell(p.Results.nR,lNN,lNM);
post = cell(p.Results.nR,lNN,lNM);
AUC = cell(p.Results.nR,lNN,lNM);

for r = 1:p.Results.nR
    disp(['Running repeat ' num2str(r) '/' num2str(p.Results.nR)]);
    
    for n = 1:lNN
        
        % Select samples
        switch p.Results.ssb
            case 'nhg'
                ix = ssb_nhg(D,y,p.Results.nN(n), 'loc', 'edge', 'type', 'T', 'viz', p.Results.viz);
            case 'nn'
                ix = ssb_nn(D,y,p.Results.nN(n), 'viz', p.Results.viz);
            case 'ax'
                ix = ssb_ax(D,y,p.Results.nN(n), 'viz', p.Results.viz);
            case 'ur'
                ix = ssb_ur(D,y,p.Results.nN(n), 'viz', p.Results.viz);
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
                Lambda = [0 10.^[-6:1:3]];
                R_la = zeros(1,length(Lambda));
                for la = 1:length(Lambda)
                    
                    % Split folds
                    ixFo = randsample(1:p.Results.nF, p.Results.nN(n), true);
                    for f = 1:p.Results.nF
                        
                        switch p.Results.clf
                            case 'tcp-ls'
                                % Train on included folds
                                theta_f = tcp_ls(X(ixFo~=f,:),yX(ixFo~=f),Z(ixnM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', Lambda(la), 'lr', p.Results.lr);
                                
                                % Evaluate on held-out source folds (MSE)
                                R_la(la) = mean(([X(ixFo==f,:) ones(length(ixFo==f),1)]*theta_f.mcpl - yX(ixFo==f)).^2);
                            case 'tcp-lda'
                                % Train on included folds
                                theta_f = tcp_lda(X(ixFo~=f,:),yX(ixFo~=f),Z(ixnM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', Lambda(la), 'lr', p.Results.lr);
                                
                                % Evaluate on held-out source folds (-ALL)
                                R_la(la) = R_la(la) - sum(sum(ll_lda(theta_f.mcpl{1}, theta_f.mcpl{2}, theta_f.mcpl{3}, X(ixFo==f,:), yX(ixFo==f)),2),1)./sum(ixFo==f);
                            case 'tcp-qda'
                                % Train on included folds
                                theta_f = tcp_qda(X(ixFo~=f,:),yX(ixFo~=f),Z(ixnM,:),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', Lambda(la), 'lr', p.Results.lr);
                                
                                % Evaluate on held-out source folds (-ALL)
                                R_la(la) = R_la(la) - sum(sum(ll_qda(theta_f.mcpl{1}, theta_f.mcpl{2}, theta_f.mcpl{3}, X(ixFo==f,:), yX(ixFo==f)),2),1)./sum(ixFo==f);
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
                    
                    % Measure on full set
                    R{r,n,m}.tcp_a = mean((D*theta{r,n,m}.tcp - y).^2,1);
                    R{r,n,m}.ref_a = mean((D*theta{r,n,m}.ref - y).^2,1);
                    R{r,n,m}.orc_a = mean((D*theta{r,n,m}.orc - y).^2,1);
                    
                    % Posteriors
                    post{r,n,m}.tcp_a = exp(D*theta{r,n,m}.tcp)./(exp(-D*theta{r,n,m}.tcp) + exp(D*theta{r,n,m}.tcp));
                    post{r,n,m}.ref_a = exp(D*theta{r,n,m}.ref)./(exp(-D*theta{r,n,m}.ref) + exp(D*theta{r,n,m}.ref));
                    post{r,n,m}.orc_a = exp(D*theta{r,n,m}.orc)./(exp(-D*theta{r,n,m}.orc) + exp(D*theta{r,n,m}.orc));
                    
                    % Predictions
                    pred{r,n,m}.tcp_a = sign(D*theta{r,n,m}.tcp);
                    pred{r,n,m}.ref_a = sign(D*theta{r,n,m}.ref);
                    pred{r,n,m}.orc_a = sign(D*theta{r,n,m}.orc);
                    
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
                    
                    % Measure on full set
                    R{r,n,m}.tcp_a = mean(sum(ll_lda(theta{r,n,m}.tcp{1},theta{r,n,m}.tcp{2},theta{r,n,m}.tcp{3},D,y),2),1);
                    R{r,n,m}.ref_a = mean(sum(ll_lda(theta{r,n,m}.ref{1},theta{r,n,m}.ref{2},theta{r,n,m}.ref{3},D,y),2),1);
                    R{r,n,m}.orc_a = mean(sum(ll_lda(theta{r,n,m}.orc{1},theta{r,n,m}.orc{2},theta{r,n,m}.orc{3},D,y),2),1);
                    [e{r,n,m}.tcp_a, pred{r,n,m}.tcp_a, post{r,n,m}.tcp_a, AUC{r,n,m}.tcp_a] = lda_err(theta{r,n,m}.tcp{1},theta{r,n,m}.tcp{2},theta{r,n,m}.tcp{3},D,y);
                    [e{r,n,m}.ref_a, pred{r,n,m}.ref_a, post{r,n,m}.ref_a, AUC{r,n,m}.ref_a] = lda_err(theta{r,n,m}.ref{1},theta{r,n,m}.ref{2},theta{r,n,m}.ref{3},D,y);
                    [e{r,n,m}.orc_a, pred{r,n,m}.orc_a, post{r,n,m}.orc_a, AUC{r,n,m}.orc_a] = lda_err(theta{r,n,m}.orc{1},theta{r,n,m}.orc{2},theta{r,n,m}.orc{3},D,y);
                case 'tcp-qda'
                    % Train on source set and test on target set
                    [theta{r,n,m},q{r,n,m},R{r,n,m},e{r,n,m},pred{r,n,m},post{r,n,m},AUC{r,n,m}] = tcp_qda(X,yX,Z(ixnM,:), 'yZ', yZ(ixnM),'maxIter', p.Results.maxIter, 'xTol', p.Results.xTol, 'alpha', p.Results.alpha, 'lambda', lambda, 'lr', p.Results.lr);
                    
                    % Measure on full set
                    R{r,n,m}.tcp_a = mean(sum(ll_qda(theta{r,n,m}.tcp{1},theta{r,n,m}.tcp{2},theta{r,n,m}.tcp{3},D,y),2),1);
                    R{r,n,m}.ref_a = mean(sum(ll_qda(theta{r,n,m}.ref{1},theta{r,n,m}.ref{2},theta{r,n,m}.ref{3},D,y),2),1);
                    R{r,n,m}.orc_a = mean(sum(ll_qda(theta{r,n,m}.orc{1},theta{r,n,m}.orc{2},theta{r,n,m}.orc{3},D,y),2),1);
                    [e{r,n,m}.tcp_a, pred{r,n,m}.tcp_a, post{r,n,m}.tcp_a, AUC{r,n,m}.tcp_a] = qda_err(theta{r,n,m}.tcp{1},theta{r,n,m}.tcp{2},theta{r,n,m}.tcp{3},D,y);
                    [e{r,n,m}.ref_a, pred{r,n,m}.ref_a, post{r,n,m}.ref_a, AUC{r,n,m}.ref_a] = qda_err(theta{r,n,m}.ref{1},theta{r,n,m}.ref{2},theta{r,n,m}.ref{3},D,y);
                    [e{r,n,m}.orc_a, pred{r,n,m}.orc_a, post{r,n,m}.orc_a, AUC{r,n,m}.orc_a] = qda_err(theta{r,n,m}.orc{1},theta{r,n,m}.orc{2},theta{r,n,m}.orc{3},D,y);
            end
        end
    end
end

% Write results
di = 1; while exist(['results_' p.Results.clf '_' p.Results.svnm num2str(di) '.mat'], 'file')~=0; di = di+1; end
fn = ['results_' p.Results.clf '_' p.Results.svnm num2str(di)];
disp(['Done. Writing to ' fn]);
save(fn, 'theta', 'q','R', 'e', 'post', 'AUC', 'lambda', 'p');

end
