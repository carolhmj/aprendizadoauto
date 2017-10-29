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

%Important variables
M = columns(mappedData); %Number of attributes + classification
alpha = 0.01;
epochs = 1000;
lambda = [0 0.01 0.25];

for i=1:columns(lambda) %Iterate over all the values of lambda
  X = mappedData(:,1:(M-1))';
  Y = mappedData(:,M);
  W = rand(M-1,1);
  
  W = stochasticLogisticRegression(W, X, Y, alpha, epochs, lambda(i));
  figure(1+i);
  plotDecisionBoundary(W,X',Y); 
end 

