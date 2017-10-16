close all;
clc;
clear;

data = load('ex2data2.txt');
data(:,1:2) = columnsToRange0_1(data(:,1:2));

%First, plot the data - we can see it's not linearly separable
figure(1);
hold on;
scatter(data(:,1), data(:,2), 3, data(:,3));

%Then, we'll use the mapFeature to generate new features,
%and create a new data matrix with the new features and 
%the category data
mappedData = [mapFeature(data(:,1), data(:,2)) data(:,3)];

%Separating the data in training and test sets
train = mappedData(1:80,:);
test = mappedData(81:118,:);
%Important variables
N = rows(train); %Number of data points
Nt = rows(test);
M = columns(train); %Number of attributes + classification
alpha = 0.01;
epochs = 1000;
lambda = [0 0.01 0.25];

avg_error_lambda = zeros(columns(lambda), 1);

for i=1:columns(lambda) %Iterate over all the values of lambda
  X = [ones(1, N) ; train(:,1:(M-1))'];
  Xt = [ones(1, Nt) ; test(:,1:(M-1))'];
  Y = train(:,M);
  Yt = test(:,M);
  W = ones(M,1);
  for j=1:epochs %Repeat for all the epochs
    [W, errors] = stochasticLogisticRegressionWithErrors(W, X, Y, Xt, Yt, alpha, epochs, lambda(i));
    avg_error_lambda(i) = mean(errors); 
  end
end

figure(2);
plot(avg_error_lambda);  