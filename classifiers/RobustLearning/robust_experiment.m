% This function is to run one trial of experiment
% Input: Xtr - (t x n) training data matrix, each row is an example
%        Xte - (te x n) test data matrix
%        ytr - (t x 1) training labels, +-1 if classification
%        yte - (te x 1) test labels
%        options - options for experiment
%          -- beta - (scalar) parameter for regularization on model theta
%          -- sigma - (scalar) kernel width for gaussian adversary
%          -- gamma - (scalar) parameter for soft moment matching constraints 
%          -- type - (character) 'R' for regression, 'C' for classification
%          -- kernel - kernel type for model theta
%          -- learner_sigma - (scalar) kernel parameter for model theta
%          -- B - (scalar) bound on alpha
%          -- maxIter - (scalar) maximum # of iterations for solvers
%          -- w - (t x 1) specified weights for learning
% Output: lossAd - adversarial losses
%           -- robTr - adversarial training loss
%           -- robTe - adversarial test loss
%         loss - losses
%           -- robTe - robust test loss of RCSA
%           -- nonRobTr - regular training loss
%           -- nonRobTe - regular test loss
%           -- other methods
%         errAd - adversarial errors
%         err - errors
%         theta - learned models
%           -- rob - robust model (or RCSA model)
%           -- nonRob - regular model
%           -- other methods
%         sigma - parameters chosen by algorithms
%         weight - weights for points
%           -- robTr - weights for training points
%           -- robTe - weights for test points
%           -- other methods
%         alpha - alphas chosen by the adversary
%           -- rob - alphas chosen in robust learning
%           -- nonRob - alphas chosen by regular learning
% Last modified: July 28, 2014
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function [lossAd, loss, errAd, err, theta, sigma, weight, alpha] =...
    robust_experiment(Xtr, Xte, ytr, yte, options)
    options.maxIter = 500;
    options.B = 5;
    options.tol = 1e-4;
    [t, n] = size(Xtr);

    % CV to choose kernel width sigma and regularizer beta
    options.gamma = 0;
    optLinearC = options.C;
    options.beta = 1/(t*optLinearC);

    % Choose proper kernel width sigma if not specified
    if(~isfield(options, 'sigma'))
        Dist = pdist2(Xtr, Xtr, 'euclidean', 'Smallest', floor(t/5))';
        options.sigma = mean(Dist(:,end));
    end
    sigma.rob = options.sigma;

    if(options.useGamma)
        Xref = Xte;
        % Heuristic gamma
        options.gamma = 0; % dummy
        [~,~,~,minErrL] = learn(Xtr, Xte, Xref, ytr, yte, options);
        Ktr_tr = gausskernel(Xtr, Xtr, options.sigma);
        Ktr_ref = gausskernel(Xtr, Xref, options.sigma);
        Kref_ref = gausskernel(Xref, Xref, options.sigma);
        k = size(Xte,1);
        difMean = sum(sum(Ktr_tr))/(t*t) + sum(sum(Kref_ref))/(k*k) -...
            (2/(t*k))*sum(sum(Ktr_ref));
        options.gamma = 2*minErrL/difMean;
    else
        options.gamma = 0;
        Xref = Xtr; % training points as reference
        if t > 500
            rp = randperm(t);
            Xref = Xtr(rp(1:500),:);
        end
    end

    % Running robust and non-robust learning
%     display('Running robust learning...');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     options.I = kmeans(Xtr, 2);
%     options.I = full(sparse(1:t,options.I,1));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [theta.rob, alpha.rob, weight, ~, lossAd.robTr, lossAd.robTe, loss.robTe,...
        errAd.robTr, errAd.robTe, err.robTe] =...
        robust_learn(Xtr, Xte, Xref, ytr, yte, options);
%     [theta.nonRob, alpha.nonRob, loss.nonRobTr, loss.nonRobTe, lossAd.nonRobTr, lossAd.nonRobTe,...
%         err.nonRobTr, err.nonRobTe, errAd.robTr, errAd.robTe] =...
%         learn(Xtr, Xte, Xref, ytr, yte, options);
%     if(options.useGamma) % compare different methods
%         % KMM
%         sigma.KMM = sqrt(n/2);
%         weight.KMM = betaKMM(Xtr, [], Xte, [], sigma.KMM, 0, 1);
%         options.w = (weight.KMM)./t;
%         [theta.KMM, ~, ~, loss.KMM] = learn(Xtr, Xte, Xref, ytr, yte, options);
%         % KLIEP
%         [weight.KLIEP,~,sigma.KLIEP] = KLIEP(Xtr', Xte',[]);
%         options.sigma = sigma.KLIEP;
%         weight.KLIEP = weight.KLIEP';
%         options.w = (weight.KLIEP)./t;
%         [theta.KLIEP, ~, ~, loss.KLIEP] = learn(Xtr, Xte, Xref, ytr, yte, options);
%         % Cluster
%         weight.Clust = compClusterWeights(Xtr, Xte, 5);
%         options.w = (weight.Clust)./sum(weight.Clust);
%         [theta.Clust, ~, ~, loss.Clust] = learn(Xtr, Xte, Xref, ytr, yte, options);
%         % RuLSIF
%         [~,weight.RuLSIF] = RuLSIF(Xte',Xtr',[],0.05);
%         weight.RuLSIF = weight.RuLSIF';
%         options.w = (weight.RuLSIF)./sum(weight.RuLSIF);
%         [theta.RuLSIF, ~, ~, loss.RuLSIF] = learn(Xtr, Xte, Xref, ytr, yte, options);
%     end
end
