function [ix] = ssb_nn(D,y,N,varargin)
% Sample selection bias using nearest-neighbours

% Shuffle seed
rng('shuffle')

% Parse hyperparameters
p = inputParser;
addOptional(p, 'viz', false);
parse(p, varargin{:});

[M,~] = size(D);
lab = unique(y);
K = numel(lab);

% Step 1: randomly sample K objects from each class
seedPoint = zeros(1,K);
for k = 1:K
    seedPoint(k) = randsample(find(y==lab(k)),1);
end

% Step 2: Find N/(nP*K) nearest neigbours from each point
nSubPop = min(M,N.*K);
nnIndex = zeros(nSubPop-1, K);
for k = 1:K
    dist = pdist2(D(seedPoint(k),:), D, 'euclidean');
    [~,ixDist] = sort(dist, 'ascend');
    nnIndex(:,k) = ixDist(2:nSubPop);
end

% Step 3: Uniformly sample N objects from subpopulation
ix = [seedPoint(:); randsample(nnIndex(:), N-K, false)];

if p.Results.viz
    X = D(ix,:);
    yX = y(ix);
    
    figure()
    plot(D(:,1),D(:,2),'k.', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    hold on;
    plot(X(yX==1,1),X(yX==1,2),'rs', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    plot(X(yX==2,1),X(yX==2,2),'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    plot(D(seedPoint(1),1),D(seedPoint(1),2),'gh','MarkerSize', 9, 'MarkerFacecolor', 'g');
    plot(D(seedPoint(2),1),D(seedPoint(2),2),'gh','MarkerSize', 9, 'MarkerFacecolor', 'g');
    xlabel('PC1');
    ylabel('PC2');
    set(gcf, 'Color', 'w','Position', [100 100 600 300]);
    set(gca, 'FontSize', 15, 'FontWeight', 'bold');
    legend({'z', 'x|y=1', 'x|y=2', 'seed'});
    export_fig(gcf, 'ssb_nn.eps');
end

end


