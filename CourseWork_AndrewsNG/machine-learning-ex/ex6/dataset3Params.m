function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 100;
sigma = 0.3;
Errp = 100000;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
Ct = [0.01;0.03;0.1;0.3;1;3;10;30];
Sigmat = [0.01;0.03;0.1;0.3;1;3;10;30];
Err = zeros(length(Ct),length(Sigmat));
for i = 1:length(Ct)
    for j=1:length(Sigmat)
        model= svmTrain(X, y, Ct(i), @(x1, x2) gaussianKernel(x1, x2, Sigmat(j))); 
        predictions = svmPredict(model, Xval);
        Err(i,j)= mean(double(predictions ~= yval));
        if(abs(Err(i,j)) <= abs(Errp))
            C = Ct(i);
            sigma = Sigmat(j); 
            Errp=Err(i,j);
        end
    end
end


% =========================================================================

end
