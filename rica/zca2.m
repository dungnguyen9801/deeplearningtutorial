function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenision of z = x.

avg = mean(x, 1);     % Compute the mean pixel intensity value separately for each patch. 
x = x - repmat(avg, size(x, 1), 1); %avg = mean(x, 1);
sigma = x * x' / size(x, 2);
[U,S,V] = svd(sigma);
Z = xZCAwhite = U * diag(1./sqrt(diag(S) + epsilon)) * U' * x;
end
