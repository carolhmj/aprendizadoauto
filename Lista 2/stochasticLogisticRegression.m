function W = stochasticLogisticRegression(W, X, Y, alpha, epochs, lambda)
  N = columns(X);
  for j=1:epochs
    %Permutate the inputs
    perm = randperm(N);
    X = X(:, perm);
    Y = Y(perm);
    %Calculate the coefficients
    for i=1:N
      error = Y(i)-logistic(W'*X(:,i));
      %This W_update is just a way to zero the first coefficient
      %so it isn't updated along with the lambda
      W_update = W;
      W_update(0) = 0;
      W = W + alpha*(error*X(:,i) - lambda*W_update);
      end
  end
end

function out = logistic(x)
  out = 1/(1+exp(-x));
end