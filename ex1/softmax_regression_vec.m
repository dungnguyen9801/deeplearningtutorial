function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  A = exp(theta'*X);
  A = [A;ones(1,size(A,2))];
  A = bsxfun(@rdivide,A,sum(A));
  B = A';
  I = sub2ind(size(B), 1:size(B,1), y);
  f = -sum(log(B(I)));

  C = zeros(size(B));
  C(I) = 1;
  C = B-C;
  G = X*C;
  g = G(:,1:size(theta,2))(:);
