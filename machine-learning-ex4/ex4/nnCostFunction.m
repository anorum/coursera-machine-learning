function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2);         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%change Y to vectorize output. so that if y = 5 then the output will be 
%[0 0 0 0 1 0 0 0 0 0].
%Setting it this way makes Y a 4000 * 10 matrix. 5000 corresponding to the
%number of training examples and 10 to the num of labels. 
 
map = eye(num_labels); %First Make a 10x10 matrix with 1 for each 1-10 spots

Y = zeros(m,num_labels); % Set Y to a empty matrix of 5000 x 10 dimension

%Next were going to loop through each row in y and and use our map to 
%pick the index and set that as Y. For Example if y = 5 then it will pull the
% 5th index in the map which is 0 0 0 0 1 0 0 0 0 0 . 
for i=1:m
  Y(i,:) = [map(y(i),:)];
endfor

X = [ones(m,1) X]; %Adding in our Bias Node to our data

%Next we are going to forward propogate to get our output of our neural network.
%Once we have our output we can then calculate the cost using cost function
% The cost function will be J = -(1/m)*sum(sum(Y.*log(h)+(1-Y).*log(1-h))) 
% This cost function is taking the average of the sum of the error for each of
% our output nodes for each training example. take the sum of each output (K) 
% for each training example m then divide by the total training examples. 

%set a1 = X
a1 = X;
%Use a1 (or X) to calculate z2. z2 is the values of of each training multiplied
%by the weights in Theta1 . Keep in mind that we have 5000 training examples and
% we have 401 features (including bias). We have 25 nodes in the hidden layer. 

%Theta1 is a 25 x 401 matrix. Remember each node in the hidden layer has a weight
% for each feature in the input layer. So each row is the weights for each feature
% in the corresponding node. I.e. row 1 = node 1. In order to calculate our values
% for layer2 we will need to take our 5000 training examples and multiply each feature
% by the corresponding weight for each node. 

%Think about this as training 25 separate logistics regression models. if we take our
% Theta and transpose it, it will give us a 401 x 25 matrix and we can multiple
% a1 * Theta1' which will output in a 5000 x 25 matrix the output for each node
% out of our 5000 training examples. 
z2 = a1 * Theta1';
a2 = sigmoid(z2); % Next we take the sigmoid of that output to force the values between 0 and 1

%we need to add in our bias unit for layer2. we will add in a vector of 5000 ones.
%this will make a2 into a 5000 x 26 matrix
a2 = [ones(size(a2,1),1) a2];

%We do the same thing for the output layer. Except this time we use a2 as our input values and
% Theta2 as our weights. Theta2 is a 10 x 26 matrix. Same deal here, each row is a node and each
% column is the weight that corresponds to a node. I.e. row1, column1 corresponds to node1 and each
% of the weights inside it. 

z3 = a2 * Theta2';
a3 = sigmoid(z3); 
%a3 is going to be a 5000 x 10 matrix. Each row corresponding to a training example
%and each column corresponding to a nodes output. i.e. the prediction for each output (1-10)
h = a3; %a3 is our final output layer so we can set this equal to h. 

%Now we have our predictions made with our initial randomized Thetas. So now we
%can compute our cost function. 
J = -(1/m)*sum(sum(Y.*log(h)+(1-Y).*log(1-h)));


%Now lets add our regularization function. Regularization will help us not overfit
%our model to our training data. Our regularization function logic works like this
% L = Total Number of Layers
% sl = number of units (not counting bias unit) in layer l 
% K = number of output units/classes

%So if we use our model as an example; L = 3 , s1 = 400 s2 = 25 s3 = 10 , K = 10

%Regularization does the following, 
% We need to square each weight, in each node, in each layer and then sum it all.
% Then we need to multiply it by lambda/2m . We can take advantage of a few things;
% The number of rows in our current layer is equal to the number of columns in the next layer (excluding bias)
% The number of columns in our current layer is equal to number of nodes in current layer

reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).**2))+sum(sum(Theta2(:,2:end).**2)));

J = J + reg;

%Now that we have our regularized cost function we need to perform back propogation
%to calculate the gradients for each Theta in each layer. So in order to do this
%we need to compute the partial deritive of each Theta(i,j) in each layer.
%So by the time were done computing each of the gradients for each of the weights
% in each of the nodes in each of the layers we should have (25*401 + 10*26) 10285 gradients


%For each of our training rows we are going to perform forward propogation to get our predictions
%We are then going to perform backpropgation to compute the gradient for each of our Thetas.

%We need to send our errors back to Layer1 for us to run our optimization function

%ALL NOTES BELOW ARE WRONG. I WENT AHEAD AND VECTORIZED THE SOLUTION

 a1 = X; % (1, 401) Matrix
 z2 = a1*Theta1'; % [1x401] * [401x25]
 a2 =  sigmoid(z2); % a2 is a [1x25] Matrix These are the predictions for this row in this layer
 a2 = [ones(size(a2,1),1) a2]; %Add bias unit [1x26]
 z3 = a2*Theta2'; % Multiply a [1x26] * [26x10]
 a3 = sigmoid(z3); %output is a [1x10] matrix of our weights
 
 %think of delta as errors
 d3 = a3 - Y; %this is the error of our prediction - the actual value of Y for this training example;
 % a3 is a [1x10] - [1x10] so outcome is a [1x10] matrix

 %Theta2 is a [10x26] an d3 is a [1x10] . We want our outcome to be each Theta times the delta for each node
 %Our total nodes is 26 so we multiple d3*Theta2 to get a [1x26] matrix. z2 is a [1x25] so need to add bias unit back in
 z2 = [ones(size(z2,1),1) z2];  %Need to add the bias unit because we did not do it above for z2; 
 d2 = (d3*Theta2).*sigmoidGradient(z2); 
 d2 = d2(:,2:end);
%No delta1. That is our feature set so no error associated with it. 
%Ignoring Regularization
%a1 is a [1x401] matrix and d2 is a [1x26] so outcome [401x1]*[1x25] = [401x25]
%a2 is a [1x26] matrix and d3 is a [1x10] so outcome [26x1]*[1x10] = [26x10]

 Theta1_grad = (a1'*d2);
 Theta2_grad = (a2'*d3);
 

 Theta1_grad = (Theta1_grad)';
 Theta2_grad = (Theta2_grad)';
 
 %Regularizing down below. This is kind of awesome so pay attention bro
 % So we want to multiply by lambda for all Thetas except for the bias weights
 % Theta1_grad is 25x401 which represents each of our gradients; Theta1 is also 25x401
 % So what we are doing below is setting the first column which corresponds to 
 % Theta1(i,0) 0 is the j here. to all zeros so when lambda is multiplied by it it will auto go to zero 
 Theta1_grad = (1/m)*(Theta1_grad + lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)]);
 Theta2_grad = (1/m)*(Theta2_grad + lambda*[zeros(size(Theta2,1),1) Theta2(:,2:end)]);
 

 
 %Need to explore why need to transpose these at the end. Must be a format
 %thing for our vectors that I will need to get used to
 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
