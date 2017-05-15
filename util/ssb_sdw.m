function [ix] = ssb_sdw(D,y,N,varargin)
% Sample selection bias using seed-distance weighing

% Shuffle seed
rng('shuffle')

% Parse hyperparameters
p = inputParser;
addOptional(p, 'seedLoc', 'min');
addOptional(p, 'viz', false);
addOptional(p, 'save', false);
parse(p, varargin{:});

% Shapes
[M,F] = size(D);

% Labels
lab = unique(y);
K = numel(lab);

% Priors
priors = zeros(1,K);
for k = 1:K
    priors(k) = mean(y==lab(k));
end

% Take a seed for each class
seed = zeros(1,K);
switch p.Results.seedLoc
    case 'min'
        norms = sqrt(sum(D.^2,2));
        for k = 1:K
            minNorm = min(norms(y==lab(k)));
            seed(k) = datasample(find(norms==minNorm), 1);
        end        
    otherwise
        for k = 1:K
            seed(k) = randsample(find(y==lab(k)),1);
        end
end

% Compute distances from seeds (negative exponential of Mahalanobis dist)
S = cov(D);
dist = zeros(K,M);
for k = 1:K
    for m = 1:M
        dist(k,m) = exp(-bsxfun(@minus, D(seed(k),:), D(m,:))/S*bsxfun(@minus, D(seed(k),:), D(m,:))');
    end
end

% Sample points from each class seed according to priors
index = [];
for k = 1:K
    yk = find(y==lab(k));
    Nk = round((N-K).*priors(k));
    index = [index; datasample(yk, Nk, 'Replace', false, 'Weights', dist(k,yk))];
end

% Include seeds in index
ix = [seed(:); index(:)];

if p.Results.viz
    
    [~,PC,~] = pca(D);
    X = PC(ix,:);
    yX = y(ix);
    
    figure()
    plot(PC(:,1),PC(:,2),'k.', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    hold on;
    plot(X(yX==lab(1),1),X(yX==lab(1),2),'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    plot(X(yX==lab(2),1),X(yX==lab(2),2),'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    plot(PC(seed(1),1),PC(seed(1),2),'gh','MarkerSize', 9, 'MarkerFacecolor', 'g');
    plot(PC(seed(2),1),PC(seed(2),2),'gh','MarkerSize', 9, 'MarkerFacecolor', 'g');
    xlabel('PC1');
    ylabel('PC2');
    set(gcf, 'Color', 'w','Position', [100 100 600 300]);
    set(gca, 'FontSize', 15, 'FontWeight', 'bold');
    legend({'z', 'x|y=1', 'x|y=2', 'seed'});
    
    if p.Results.save
        export_fig(gcf, 'ssb_sdw.eps');
    end
end

end


