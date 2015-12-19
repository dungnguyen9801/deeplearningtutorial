%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

sparse_epsilon = .01;
Wx = W*x;
abs_Wx = sqrt(Wx.^2 + sparse_epsilon);
cost_sparse = sum(sum(abs_Wx)) * params.lambda;
error_Wx = Wx./abs_Wx * params.lambda; % derive of sqrt(a^2+e) is a/sqrt(a^2+e)
Wgrad_sparse = error_Wx * x';

Wx = W*x;
WWx = W' * Wx;
cost_recon = .5*sum(sum((WWx - x).^2));
error_WWx = WWx - x;
error_Wx = W*error_WWx;
Wgrad_Wt = Wx*error_WWx';
error_x = W'*error_Wx;
Wgrad_W = error_Wx*x';
Wgrad_recon = Wgrad_W + Wgrad_Wt;
%Wgrad_recon = W*(W'*W*x-x)*x' + (W*x)*(W'*W*x-x)';

cost = cost_sparse+ cost_recon;
Wgrad = Wgrad_sparse + Wgrad_recon;

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
