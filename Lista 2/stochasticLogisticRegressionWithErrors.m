%W : coefficients
%X : attributes for training set
%Y : category for training set
%Xt : attributes for test set
%Yt : category for test set
%alpha : training alpha 
%epochs : # of epochs for training
%lambda : lambda for regularization
function [W, errors] = stochasticLogisticRegressionWithErrors(W, X, Y, Xt, Yt, alpha, epochs, lambda)
  %Number of data points from training and test sets
  N = columns(X);
  Nt = columns(Xt);
  %Vector of errors over epochs
  errors = zeros(epochs,1);
  for j=1:epochs
    %Permutate the inputs
    perm = randperm(N);
    X = X(:, perm);
    Y = Y(perm);
    %Calculate the coefficients on training set
    for i=1:N
      error = Y(i)-logistic(W'*X(:,i));
      %This W_update is just a way to zero the first coefficient
      %so it isn't updated along with the lambda
      W_update = W;
      W_update(1) = 0;
      W = W + alpha*(error*X(:,i) - lambda*W_update);
      end
    %Calculate the avg errors on test set
    for i=1:Nt
      error = Yt(i)-logistic(W'*Xt(:,i));
      errors(j) = errors(j) + error^2/Nt; %Divide error by number of points since we want the average
    end
  end
end

function out = logistic(x)
  out = 1/(1+exp(-x));
end