function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples


% X = [m n]
% y = [m 1]
% theta = [n 1]

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% =============================================================

h_x = sigmoid(X * theta);
% [m n] [n 1] => [m 1]
J = (1/m) * (-y' * log(h_x) - (1-y)' * log(1 - h_x));
J = J + (lambda / (2*m)) * sum(theta(2:end).^2);  # add the regularlization values

%Grad without regularization
grad_partial = (1/m) * (X' * (h_x -y));

%%Grad Cost Added
grad_regularization = (lambda/m) .* theta(2:end); 
% don't regularize first term of theta

grad_regularization = [0; grad_regularization];
% tack theta_0 onto the theta vector

grad = grad_partial + grad_regularization;

end
