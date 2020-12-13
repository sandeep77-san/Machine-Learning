function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
q = length(theta);
% You need to return the following variables correctly 
J = 0;
J1 =0;
J2 = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

a = zeros(m);
h = zeros(m);
a = X*theta;
h = sigmoid(a);
J1 = (1/m)*sum(-(transpose(y)*log(h))-(transpose(1-y)*log(1-h)));

J2 = sum(theta.*theta) - theta(1,1)^2;
J = J1 + (lambda/(2*m))*J2;

Err = (h-y).*X;

for i = 1:q
    if (i==1)
        grad(i,1)= (1/m)*sum(Err(:,i));
    else
        grad(i,1)= (1/m)*sum(Err(:,i)) + (lambda/m)*theta(i,1);
    end
end





% =============================================================

end
