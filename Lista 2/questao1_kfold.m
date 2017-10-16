close all;
clc;
clear;

data = load('ex2data1.txt');
%normalizing
data(:,1:2) = columnsToRange0_1(data(:,1:2));

alpha = 0.01;
epochs = 1000;

k = 5; % # of folds
W = []; % will store the coefficients
eqm = zeros(epochs, k);
% m denotes the fold that is chosen for tests. all other folds are chosen to 
% form the training set
for m=1:k 
  W(:,m) = [1; 1; 1]; %initial guess
  
  train = [];
  test = [];
  
  for n=1:k 
    if (n == m)
      test = data((n-1)*(100/k)+1:n*(100/k),:); %forming the test set 
    else
      train = [train; data((n-1)*(100/k)+1:n*(100/k),:)]; %forming the training set
    end
  end
    
    %training data
    X = [ones(1,80); train(:,1:2)'];
    Y = train(:,3);

    %test data
    Xt = [ones(1,20); test(:,1:2)'];
    Yt = test(:,3);
    
    [W(:,m), eqm(:,m)] = stochasticLogisticRegressionWithErrors(W(:,m), X, Y, Xt, Yt, alpha, epochs, 0);
    figure(m);
    plot(eqm(:,m));    
end