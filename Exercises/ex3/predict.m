function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% variable dimensions:
% X = [m x]
% Theta1 = [s x+1]
% Theta2 = [10 s+1]
% p = [m 1]

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
% =========================================================================

% compute first layer
Bias_X = [ones(size(X,1), 1) X];  % [m x+1]
X2 = sigmoid(Bias_X * Theta1');   % [m x+1] * [x+1 s] => [m s]

% compute second layer
Bias_X2 = [ones(size(X2,1), 1) X2];   % [m s+1]
X3 = sigmoid(Bias_X2 * Theta2');      % [m s+1] * [s+1 10] => [m 10]

% extract predictions based on max values per row
[val p] = max(X3, [], 2);
end
