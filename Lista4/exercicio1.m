clc;
clear;

function out = prob(covar, avg, x, dim)
  out = 1/(det(covar)^(1/2)*(2*pi)^(dim/2))*exp((-1/2)*(x-avg)'*inv(covar)*(x-avg));
end

%Load the matrices
load('DadosLista4.mat');

%Get training and test sets
%Training should be 600 - 200 from each of the 3 classes
%Test should be 900 - 300 from each of the 3 classes
%We know the first 500 examples are of class 1, etc...
%So we can divide by class

cls1 = Dados(1:500,:);
cls2 = Dados(501:1000,:);
cls3 = Dados(1001:1500,:);

%Performing first permutation
r1 = randperm(500);
r2 = randperm(500);
r3 = randperm(500);

cls1 = cls1(r1,:);
cls2 = cls1(r2,:);
cls3 = cls1(r3,:);

%Constructing vectors
trainC1 = cls1(1:200,:);
trainC2 = cls2(1:200,:);
trainC3 = cls3(1:200,:);
trN = 200;

test = [cls1(201:end,:); cls2(201:end,:); cls3(201:end,:)];
testCls = [ones(300,1); 2*ones(300,1); 3*ones(300,1)];
teN = 900;

%DQM classificator
%Calculate average for each class
avg = zeros(3,2);
for i=1:trN
  avg(1,:) = avg(1,:) + trainC1(i,:);
  avg(2,:) = avg(2,:) + trainC2(i,:);
  avg(3,:) = avg(3,:) + trainC3(i,:);
end
avg(1,:) = avg(1,:)/trN;
avg(2,:) = avg(2,:)/trN;
avg(3,:) = avg(3,:)/trN;

%Calculate covariance matrix
covar = zeros(6,2);
for i=1:trN
  covar(1:2,:) = covar(1:2,:) + (trainC1(i,:)-avg(1,:))'*(trainC1(i,:)-avg(1,:));
  covar(3:4,:) = covar(3:4,:) + (trainC2(i,:)-avg(2,:))'*(trainC2(i,:)-avg(2,:));
  covar(5:6,:) = covar(5:6,:) + (trainC3(i,:)-avg(3,:))'*(trainC3(i,:)-avg(3,:));
end
covar(1:2,:) = covar(1:2,:)/(trN-1);
covar(3:4,:) = covar(3:4,:)/(trN-1);
covar(5:6,:) = covar(5:6,:)/(trN-1);

%Calculate errors in confusion matrix for test set
conf = zeros(3);

for i=1:teN
  p = zeros(3,1);
  %Calculate probability of having these characteristics in each class
  pX_C1 = prob(covar(1:2,:), avg(1,:)', test(i,:)', 2);
  %Probability of each class is equal to 1/3
  p(1) = pX_C1*1/3;
  %Repeat for other two classes  
  pX_C2 = prob(covar(3:4,:), avg(2,:)', test(i,:)', 2);
  p(2) = pX_C2*1/3;
  pX_C3 = prob(covar(5:6,:), avg(3,:)', test(i,:)', 2);
  p(3) = pX_C3*1/3;
  
  %Select one with higest prob
  [m,cls] = max(p);
  %Add value to correct position on confusion matrix
  conf(cls,testCls(i)) = conf(cls,testCls(i)) + 1;
end  