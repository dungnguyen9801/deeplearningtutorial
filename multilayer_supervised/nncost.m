function [ costnn, gradnn, pred_prob] = nncost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)


%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
m = size(labels,1);
hidden_layer_size = ei.layer_sizes(1);
input_layer_size = ei.input_dim;
nn_params = theta;
num_labels = ei.output_dim;

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

X = data';
y = labels';

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

if po
  costnn = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  p = zeros(size(X, 1), 1);

  h1 = sigmoid([ones(m, 1) X] * Theta1');
  h2 = sigmoid([ones(m, 1) h1] * Theta2');
  pred_prob = h2';
  return;
end;

I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
Z3 = A2*Theta2';
H = A3 = sigmoid(Z3);

lambda = ei.lambda;

penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));

J = (1/m)*sum(sum((-Y).*log(H) - (1-Y).*log(1-H), 2));
J = J + penalty;

Sigma3 = A3 - Y;
Sigma2 = (Sigma3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);


Delta_1 = Sigma2'*A1;
Delta_2 = Sigma3'*A2;


Theta1_grad = Delta_1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
gradnn = [Theta1_grad(:) ; Theta2_grad(:)];
costnn = J;
end

function ret = sigmoid(Z)
  ret = 1./(1+exp(-Z));
end

function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

g = sigmoid(z).*(1 - sigmoid(z));

% =============================================================

end
