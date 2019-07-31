function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    thetat=theta;
      H = X*theta-y;
    for i=1:1:size(theta,1)
%           theta(i)=theta(i)-alpha*(1/m)*sum((X*theta-y).*X(:,i));
%         
               
     
    thetat(i)=thetat(i)-alpha*(1/m)*sum(H.*X(:,i));

    end
   theta=thetat;

    
    %下面是答案，这个地方主要我主要没考虑到的地方就是更新的同时性的问题    
%    h = X*theta;
%     temp(:,iter) = theta - ((alpha/m)*(X'*(h-y)));
%     theta = temp(:,iter);
% 
%  334302.063993 
%  100087.116006 
%  3673.548451 


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
