function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

%fprintf('Size of X: [%d %d]\n', size(X))         % [m n]
%fprintf('Size of y: [%d %d]\n', size(y))         % [m 1]
%fprintf('Size of theta: [%d %d]\n', size(theta)) % [n 1]

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% =========================================================================

J_direct =  1 / (2 * m) * sum((X * theta - y) .^ 2);  % [m n] * [n 1] => [m 1] - [m 1]
J_reg = lambda / (2*m) * sum(theta(2:end) .^ 2);
J = J_direct + J_reg;

h_x = X * theta;       % [m n] * [n 1] => [m 1]
grad_direct = 1 / m * X' * (h_x - y);
grad_reg = [0; lambda / m * theta(2:end)];
grad = grad_direct+ grad_reg;
grad = grad(:);
end
