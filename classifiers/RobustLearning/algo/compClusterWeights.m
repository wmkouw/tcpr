% This function is a toy implementation of cluster-based reweighing method
% See also Cortes et al, Sample selection bias correction theory
% Input: Xtr - (t x n) training data matrix, each row is an example
%        Xte - (te x n) test data matrix
%        k - (scalar) the # of clusters
% Output: weight - weights for points
% Last modified: July 28, 2014
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function weight = compClusterWeights(Xtr, Xte, k)
    X = [Xtr; Xte];
    t = size(Xtr, 1);
    te = size(Xte, 1);
    idx = kmeans(X, k);
    idxTr = idx(1:t);
    idxTe = idx((t+1):(t+te));
    cWeights = zeros(k,1);
    for i = 1:k
        cWeights(i) = sum(idxTe==i)/sum(idxTr==i);
    end
    weight = cWeights(idxTr);
end
