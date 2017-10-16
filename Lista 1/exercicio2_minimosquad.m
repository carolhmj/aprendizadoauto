clc;
clear;

data = load('ex1data2.txt');
n = 47;
X = [ones(n,1) data(:,1:2)];
Y = data(:,3);

W = inv(X'*X)*X'*Y;