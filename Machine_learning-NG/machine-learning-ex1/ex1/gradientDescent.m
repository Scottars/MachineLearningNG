function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
J1=0;
J2=0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    J1=0;
J2=0;
    for i=1:1:m
        
        J1= J1 + (theta'*X(i,:)'-y(i))*X(i,1);
        J2= J2 + (theta'*X(i,:)'-y(i))*X(i,2);
        
    end
%       H = X*theta-y;
%    J1=sum(H.*X(:,1))
%     J2=sum(H.*X(:,2))
%     
    
         theta(1)=theta(1)-alpha*J1/m;
         theta(2)=theta(2)-alpha*J2/m;
         
       
%        H = X*theta-y;
%     theta(1)=theta(1)-alpha*(1/m)*sum(H.*X(:,1));
%     theta(2)=theta(2)-alpha*(1/m)*sum(H.*X(:,2));

% 
%      temp=X'*(X*theta-y);
%      theta=theta-1/m*alpha*temp;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
