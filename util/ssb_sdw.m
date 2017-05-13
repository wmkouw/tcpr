function [ix] = ssb_sdw(D,y,N,varargin)
% Sample selection bias using seedpoint-distance weighing

% Shuffle seed
rng('shuffle')

% Parse hyperparameters
p = inputParser;
addOptional(p, 'viz', false);
addOptional(p, 'save', false);
parse(p, varargin{:});

[M,F] = size(D);
lab = unique(y);
K = numel(lab);

% Step 1: randomly sample K objects from each class
seed = zeros(1,K);
for k = 1:K
    seed(k) = randsample(find(y==lab(k)),1);
end

% Compute distances from seeds
S = eye(F);
dist = zeros(K,M);
for k = 1:K
    for m = 1:M
        dist(k,m) = exp(-bsxfun(@minus, D(seed(k),:), D(m,:))/S*bsxfun(@minus, D(seed(k),:), D(m,:))');
    end
end

% Step 2: Find N/K points from the seed
Nk = floor((N-K)./K);
index = zeros(Nk, K);
for k = 1:K
    index(:,k) = datasample(1:M, Nk, 'Replace', false, 'Weights', dist(k,:));
end

% Step 3: Uniformly sample N objects from subpopulation
ix = [seed(:); index(:)];

if p.Results.viz
    X = D(ix,:);
    yX = y(ix);
    
    figure()
    plot(D(:,1),D(:,2),'k.', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    hold on;
    plot(X(yX==1,1),X(yX==1,2),'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    plot(X(yX==2,1),X(yX==2,2),'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    plot(D(seed(1),1),D(seed(1),2),'gh','MarkerSize', 9, 'MarkerFacecolor', 'g');
    plot(D(seed(2),1),D(seed(2),2),'gh','MarkerSize', 9, 'MarkerFacecolor', 'g');
    xlabel('PC1');
    ylabel('PC2');
    set(gcf, 'Color', 'w','Position', [100 100 600 300]);
    set(gca, 'FontSize', 15, 'FontWeight', 'bold');
    legend({'z', 'x|y=1', 'x|y=2', 'seed'});
    
    if p.Results.save
        export_fig(gcf, 'ssb_nnw.eps');
    end
end

end


