function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


z=X*theta;
ht=sigmoid(z);
%sum1=sum(-y.*log(ht))
%sum2=sum((1-y).*log(1-ht))
thetar=(theta(2:end)); % gives array from element 2 to end

Jl=-1/m*(sum(y.*log(ht))+sum((1-y).*log(1-ht))); % logistic regression
Jr=+(lambda/(2*m))*sum(thetar.^2); % normalised regression
J=Jl+Jr;


%                                   Gradient


grad0= 1/m*(X(:,1)'*(ht-y)); % gradient of theta[0]
grad1m = 1/m * X(:,2:end)'*(ht-y) + (lambda/m*thetar); % gradient of theta 1 to end

grad = [grad0;grad1m]; % to stop some matrix error


% =============================================================

end









