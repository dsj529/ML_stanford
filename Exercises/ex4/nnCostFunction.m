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
% -------------------------------------------------------------

% =========================================================================

% Step 1: Feed forward and compute J for the initial parameters Theta
% use a1...aN notation to line up with lecture notes/algorithm pattern
a1 = [ones(size(X,1), 1) X];  % [m x+1]
z2 = (a1 * Theta1');          % [m x+1] * [x+1 s] => [m s]
a2 = [ones(size(z2,1), 1) sigmoid(z2)];   % [m s+1]
a3 = h_theta = sigmoid(a2 * Theta2');     % [m s+1] * [s+1 l] => [m l]

y_mat = eye(num_labels)(y,:);   %  [l l];
%{
This line takes advantage of Octave's matrix indexing capabilities.
Multiplying the I(num_labels) matrix by the indexed values of y
sets the 1s in those locations.
%}


% compute base cost of J:
J_base = 1/m * (-sum(sum(y_mat .* log(h_theta))) - 
                 sum(sum((1 - y_mat) .* (log(1 - h_theta)))));
            
% compute regularized cost of J:
J_reg = (lambda / (2*m)) * ((sum(sum(Theta1(:, 2:end).^2))) + 
                             sum(sum(Theta2(:, 2:end).^2)));
J = J_base + J_reg;
            
% Step 2: Backpropagate error terms through the network
d3 = a3 - y_mat;                                                  % [m l]  ~ a3
d2 = (d3 * Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)];   % [m l] * [l s+1] => [m s+1] ~ a2

D1 = d2(:, 2:end)' * a1;   % [s m] * [m x+1] => [s x+1] ~ Theta1
D2 = d3' * a2;             % [l m] * [m s+1] => [l s+1] ~ Theta2

Theta1_grad = Theta1_grad + (1/m)*D1;
Theta2_grad = Theta2_grad + (1/m)*D2;

% Regularize backpropagated gradients
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
