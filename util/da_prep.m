function [X] = da_prep(X, prep)
% Function to run a number of preprocessing routines on a dataset
% Input:
%   X:      Dataset (MxN)
%   prep:   cell array of preps
%
% Output:
%   X:      Prepped set
%
% Author: Wouter Kouw
% Pattern Recognition & Bioinformatics group
% Delft University of Technology

% Check for cell type
if ischar(prep)
    prep = cellstr(prep);
end

% Data size
[N,D] = size(X);

% Length of preprocessing string
lP = length(prep);

for i = 1:lP
    if isnumeric(prep{i}); continue; end
    switch prep{i}
        case 'maxdiv'
            X = X./max(X(:));
            disp(['Normalized by maximum value: ' num2str(max(X(:)))]);
        case 'constdiv'
            X = X./prep{i+1};
            disp(['Normalized by constant: ' num2str(prep{i+1})]);
        case 'binarize'
            X(X>=0.5) = 1;
            X(X<0.5) = 0;
            disp(['Binarized the data (X>0.5=1, X<0.5=0)']);
        case 'zscore'
            X = bsxfun(@minus, X, mean(X,1));
            v = std(X,0,1);
            X = bsxfun(@rdivide, X, v);
            X(:,v==0) = 0;
            disp(['Z-scored each feature']);
        case 'minusmin'
            X = bsxfun(@minus, X, min(X, [], 1));
            disp(['Subtracted the minimum value for each feature']);
        case 'minusminsamp'
            X = bsxfun(@minus, X, min(X, [], 2));
            disp(['Subtracted the minimum value for each sample']);
        case 'tfidf'
            df = log(N ./ (sum(X>0, 1) + 1));
            X = bsxfun(@times, X, df);
            disp(['Ran tf-idf features']);
        case 'fmax'
            m = max(X, [], 1);
            X = bsxfun(@rdivide, X, m);
            X(:,m==0) = 0;
            disp(['Scaled each feature to max 1']);
        case 'fsum'
            X = bsxfun(@rdivide, X, sum(X, 1));
            disp(['Normalized each feature']);
        case 'fstd'
            v = std(X,0,1,'omitnan');
            X = bsxfun(@rdivide, X, v);
            X(:,v==0) = 0;
            disp(['Normalized feature variance to 1']);
        case 'norm_samp'
            X = bsxfun(@rdivide, X, sqrt(sum(X.^2,2)));
            disp(['Normalized each sample by its norm']);
        case 'sum_samp'
            v = sum(X,1);
            X = bsxfun(@rdivide, X, v);
            X(isnan(X)) = 0;
            disp(['Normalized each sample by its sum']);
        case 'pca100'
            [~,X,~] = pca(X, 'Algorithm', 'eig');
            disp(['Mapped onto PCA components']);
        case 'pca99'
            [~,X,V] = pca(X, 'Algorithm', 'eig');
            X = X(:,1:find(cumsum(V)>.99*sum(V),1,'first'));
            disp(['Mapped onto 99% variance PCA components']);
            disp(['Retained ' num2str(size(X,2)) ' components']);
        case 'pca95'
            [~,X,V] = pca(X, 'Algorithm', 'eig');
            X = X(:,1:find(cumsum(V)>.95*sum(V),1,'first'));
            disp(['Mapped onto 95% variance PCA components']);
            disp(['Retained ' num2str(size(X,2)) ' components']);
        case 'pca95iso'
            if issparse(X)
                disp(['Using sparse matrix PCA']);
                k = min(1000,D);
                X = bsxfun(@minus, X, mean(X,1));
                [V,E] = eigs(cov(X),k);
                [E,order] = sort(diag(E), 'descend');
                V = V(:,order);
                X = bsxfun(@rdivide, X*V, sqrt(E)');
                disp(['Mapped onto PC and normalized']);
            else
                [~,X,V] = pca(X);
                idx = find(cumsum(V)>.95*sum(V),1,'first');
                X = X(:,1:idx);
                X = bsxfun(@rdivide, X, sqrt(V(1:idx))');
                disp(['Mapped onto 95% variance PC and normalized']);
            end
            disp(['Retained ' num2str(size(X,2)) ' components']);
        case 'fsel'
            flist = prep{i+1};
            if ~isempty(setdiff(flist, 1:D))
                error(['Features selected that are not present']);
            end
            X = X(:,flist);
            disp(['Selected features ' num2str(flist)]);
        case 'fselr'
            nSel = prep{i+1};
            if nSel>D
                error(['More features requested than are present']);
            end
            flist = randsample(1:D,nSel,false);
            X = X(:,flist);
            disp(['Selected features ' num2str(flist)]);
        case 'impute0'
            X(isnan(X)) = 0;
            disp(['Impute missing values with 0']);
        case ''
            disp(['No data preprocessing']);
        otherwise
            error([prep{i} ' has not been implemented']);
    end
end


end
