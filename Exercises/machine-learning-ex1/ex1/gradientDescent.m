function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================


% vectorized algorithm
    Errs = theta' * X' - y'
    del_theta = Errs * X;
    % dimentional analyses
    % theta = [n * 1]
    % X = [m * n]
    % y = [m * 1]
    % Error term = [1 n] * [n m] = [1 m]
    % del_theta = [1 m] * [m n] = [1 n]
    theta = theta - (alpha/m) * del_theta';


%  iterative algorithm
%    Errs = X * theta -y ;
%    t0 = theta(1,1) - (alpha / m) * sum(Errs .* X(:,1));
%    t1 = theta(2,1) - (alpha / m) * sum(Errs .* X(:,2));
%    theta = [t0; t1];


    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
