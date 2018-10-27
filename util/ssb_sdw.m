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
[M,~] = size(D);

% Check for column vector y
if ~iscolumn(y); y = y'; end

% Labels
labels = unique(y);
K = numel(labels);

% Priors
priors = zeros(1,K);
for k = 1:K
    priors(k) = mean(y==labels(k));
end

% Take a seed for each class
seed = zeros(1,K);
switch p.Results.seedLoc
    case 'min'
        norms = sqrt(sum(D.^2,2));
        for k = 1:K
            minNorm = min(norms(y==labels(k)));
            seed(k) = datasample(find(norms==minNorm), 1);
        end        
    otherwise
        for k = 1:K
            seed(k) = randsample(find(y==labels(k)),1);
        end
end

% Compute distances from seeds (negative exponential of Mahalanobis dist)
S = cov(D);
distances = zeros(K,M);
for k = 1:K
    for m = 1:M
        distances(k,m) = exp(-bsxfun(@minus, D(seed(k),:), D(m,:))/S*bsxfun(@minus, D(seed(k),:), D(m,:))');
    end
end

% Sample points from each class seed according to priors
index = [];
for k = 1:K
    yk = find(y==labels(k));
    Nk = round((N-K).*priors(k));
    index = [index; datasample(yk, Nk, 'Replace', false, 'Weights', distances(k,yk))];
end

% Include seeds in index
ix = [seed(:); index(:)];

if p.Results.viz
    
    A = pca(D);
    PC = D*A;
    X = PC(ix,:);
    yX = y(ix);
    
    figure()
    plot(PC(:,1),PC(:,2),'k.', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    hold on;
    plot(X(yX==labels(1),1),X(yX==labels(1),2),'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    plot(X(yX==labels(2),1),X(yX==labels(2),2),'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    plot(PC(seed(1),1),PC(seed(1),2),'gh','MarkerSize', 9, 'MarkerFacecolor', 'g');
    plot(PC(seed(2),1),PC(seed(2),2),'gh','MarkerSize', 9, 'MarkerFacecolor', 'g');
    xlabel('PC1');
    ylabel('PC2');
    set(gcf, 'Color', 'w','Position', [0 0 1000 500]);
    set(gca, 'FontSize', 20, 'FontWeight', 'bold');
    legend({'z', 'x|y=-1', 'x|y=+1', 'seed'});
    
    if p.Results.save
        
        mkdir viz
        saveas(gcf, 'viz/ssb_sdw.png');
    end
end

end


