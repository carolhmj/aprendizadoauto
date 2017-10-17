%W : coefficients
%X : attributes for training set
%Y : category for training set
%alpha : training alpha 
%epochs : # of epochs for training
%lambda : lambda for regularization
function W = stochasticLogisticRegression(W, X, Y, alpha, epochs, lambda)
  %Number of data points from training set
  N = columns(X);
  for j=1:epochs
    %Permutate the inputs
    perm = randperm(N);
    X = X(:, perm);
    Y = Y(perm);
    %Calculate the coefficients on training set
    for i=1:N
      error = Y(i)-logistic(W'*X(:,i));
      %W'[1x496]*X[496x1] = error[1x1]
      %This W_update is just a way to zero the first coefficient
      %so it isn't updated along with the lambda
      W_update = W;
      %W_update[496x1]
      W_update(1) = 0;
      W = W + alpha*(error*X(:,i) - lambda*W_update);
      %W[496x1] = W[496x1] + alpha[1x1]*(error[1x1]*X[496x1] - lambda[1x1]-W_up[496x1])
    end
  end
end

function out = logistic(x)
  out = 1/(1+exp(-x));
end