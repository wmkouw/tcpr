function [K] = linearkernel(X1,X2,dummy)
	K = X1*X2';
end
