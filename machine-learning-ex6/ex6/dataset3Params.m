function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C_test = [.01; .03; .1; .3; 1; 3; 10; 30];
sigma_test = [.01; .03; .1; .3; 1; 3; 10; 30];
error_table = zeros(64,3);
t = 1;

for i=1:length(C_test)
  for j=1:length(sigma_test)
    model= svmTrain(X, y, C_test(i), @(x1, x2) gaussianKernel(x1, x2, sigma_test(j)));
    predictions = svmPredict(model, Xval);
    error_table(t,1) = C_test(i);
    error_table(t,2) = sigma_test(j);
    error_table(t,3) = mean(double(predictions ~= yval));
    t += 1;
  endfor
endfor

[M, I] = min(error_table(:,3))
C = error_table(I,1)
sigma = error_table(I,2)




% =========================================================================

end
