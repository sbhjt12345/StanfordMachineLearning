function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y);   % number of training examples

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    predictions = X*theta-y;
    theta2 = zeros(2,1);
    theta2(1) = alpha*sum(predictions)/m;
    theta2(2) = alpha*sum(predictions.*X(:,2))/m;
    theta = theta - theta2;

               
    %theta2 = zeros(2,1);
    %theta2(1) = theta(1) - alpha * sum(X*theta-y)/m;
    %theta2(2) = theta(2) - alpha * sum((X*theta-y).*X(:,2))/m;
    %theta = theta2;
               
               
    % ============================================================

    % Save the cost J in every iteration

    J_history(iter) = computeCost(X, y, theta);

end

end
