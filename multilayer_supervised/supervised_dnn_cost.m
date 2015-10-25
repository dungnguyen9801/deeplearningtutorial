function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

m = size(labels,1);
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
output = data; 
for d = 1:numel(stack)
  hAct{d} = output;
  output = 1./(1 + exp(-stack{d}.W * output - stack{d}.b));
end;
pred_prob = output;
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
C = zeros(size(labels,1), ei.output_dim);
I = sub2ind(size(C), 1:size(C,1), labels');
C(I) = 1;
C = C';
cost = (-sum(sum(C.*log(output) + (1-C).*log(1-output))))/m;
zderiv = (output - C);

%% compute gradients using backpropagation
for d = numHidden+1:-1:1
  gradStack{d}.W = zderiv*(hAct{d})'/m;
  gradStack{d}.W += ei.lambda*stack{d}.W/m;
  gradStack{d}.b = zderiv*ones(m,1)/m;
  cost += ei.lambda/(2*m)*(sum(sum(stack{d}.W.^2)));
  zderiv = ((stack{d}.W)'*zderiv).*(1-hAct{d}).*hAct{d};
end

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end

