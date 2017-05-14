function x = projsplx(y)
% project an n-dim vector y to the simplex Dn
% Dn = { x : x n-dim, 1 >= x >= 0, sum(x) = 1}

% (c) Xiaojing Ye
% xyex19@gmail.com
%
% Algorithm is explained as in the linked document
% http://arxiv.org/abs/1101.6081
% or
% http://ufdc.ufl.edu/IR00000353/
%
% Jan. 14, 2011.

[N,m] = size(y);
if m==1
    x = ones(N,1);
else
    S = sort(y,2,'descend');
    CS = cumsum(S,2);
    TMAX = bsxfun(@rdivide,CS-1,1:m);
    Bget = TMAX(:,1:end-1) < S(:,2:end);
    I = sum(Bget,2)+1;
    TMAX = TMAX';
    x = max(bsxfun(@plus,y,-TMAX(I+m*(0:N-1)')),0);
    
    % Correction
    qProj = (x*[-1 1]' + 1)/2;
    qProj = min(qProj,1-realmin);
    qProj = max(qProj,realmin);
    x = [1-qProj,qProj];
    x = max(realmin,min(1-realmin,x));
end
