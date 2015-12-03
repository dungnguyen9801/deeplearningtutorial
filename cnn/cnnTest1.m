%% Convolution Neural Network Unit Test

addpath ../common/;
numClasses = 1;
numFilters = 1;
filterDim = 1;
poolDim = 2;
images = zeros(2, 2, 1);
images(:, :, 1) = [4 7;1 9];
labels = zeros(1);
labels(1) = 1;
imageDim = size(images,1);
theta = [.4; .7; .8; .1];

[cost grad] = cnnCost(theta,images,labels,numClasses,...
                            filterDim,numFilters,poolDim);
numGrad = computeNumericalGradient( @(x) cnnCost(x,images,...
                            labels,numClasses,filterDim,...
                            numFilters,poolDim), theta);

% Use this to visually compare the gradients side by side
disp([numGrad grad]);

diff = norm(numGrad-grad)/norm(numGrad+grad);
% Should be small. In our implementation, these values are usually 
% less than 1e-9.
disp(diff); 

assert(diff < 1e-9,...
    'Difference too large. Check your gradient computation again');

