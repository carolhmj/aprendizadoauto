close all;
clc;
clear;

%Load matrices
load("ex3data1.mat");

%Normalize attributes
X = X./100;
%Randomize data, and separate in training, validation and test sets
[Xrow, Xcol] = size(X);
[Trow, Tcol] = size(T);

perm = randperm(Trow);
T(perm,:);
X(perm,:);

trainAttr = X(1:4000,:);
trainVal = T(1:4000,:);
trainNum = rows(trainAttr);


valAttr = X(4001:4500,:);
valVal = T(4001:4500,:);
valNum = rows(valAttr);

testAttr = X(4501:5000,:);
testVal = T(4501:5000,:);
testNum = rows(testAttr);

%Defining variables
hiddenNeurs = 20; %Number of hidden neurons
outNeurs = 10; %Number of out neurons
alpha = 0.1;

tolerancePassed = false;
prevMediumError = [];
trainError = [];
valError = [];
W = rand(hiddenNeurs,Xcol+1);
M = rand(outNeurs,hiddenNeurs+1);
firstEpoch = true;
%While the difference between the past validation error and the present
%validation error is less than the tolerance, do
while true
%for m=1:50
  %Train the weights on training set
  epochTrainingError = zeros(trainNum, 1);
  for i=1:trainNum
    %Start with hidden layer
    X_hid = [-1 ; trainAttr(i,:)']; %Construct the input vector
    U_hid = W*X_hid; %Calculate the sum of weights+attributes
    Z = logistic(U_hid); %Apply logistic function
   %Then use it as input for the output layer, repeat the same process
    Z_out = [-1 ; Z];
    U_out = M*Z_out;
    Y = logistic(U_out);
    %Calculate the error
    E = trainVal(i,:)' - Y;
    
    epochTrainingError(i) = E'*E;
    
    %Calculate the local gradients
    G_out = E.*(Y.*(1-Y)); %Output gradient
    G_hid = (Z.*(1-Z)).*(M(:,2:end)'*G_out); %Hidden layer gradient
    
    %Update weights
    M = M + alpha*G_out*Z_out'; %Output layer
    W = W + alpha*G_hid*X_hid'; %Hidden layer
  end
  %Save the training error for later
  trainError = [trainError; mean(epochTrainingError)];
  
  epochValError = zeros(valNum, 1);
  %Calculate the values on the validation set 
  for j=1:valNum
    %Start with hidden layer
    X_hid = [-1 ; valAttr(j,:)']; %Construct the input vector
    U_hid = W*X_hid; %Calculate the sum of weights+attributes
    Z = logistic(U_hid); %Apply logistic function
   %Then use it as input for the output layer, repeat the same process
    Z_out = [-1 ; Z];
    U_out = M*Z_out;
    Y = logistic(U_out);
    %Calculate the error
    E = valVal(j,:)' - Y;
    epochValError(j) = E'*E;
  end
  valError = [valError; mean(epochValError)];
  
  if (!firstEpoch)
    %If the error grows, end the training
    if (valError(end) - valError(end-1) > 0)
      break;
    end 
  end
  firstEpoch = false;  
end

%Draw graph to show errors
figure(1);
plot(trainError,'r');
hold on;
plot(valError,'b');
xlabel('Epochs');
ylabel('EQM');
h = legend ({"Training Error"}, "Validation Error");
legend (h, "location", "northeastoutside");

%Get classification error for test set
epochTestError = zeros(testNum, 1);
%Calculate the values on the test set 
for k=1:testNum
 %Start with hidden layer
 X_hid = [-1 ; testAttr(k,:)']; %Construct the input vector
 U_hid = W*X_hid; %Calculate the sum of weights+attributes
 Z = logistic(U_hid); %Apply logistic function
 %Then use it as input for the output layer, repeat the same process
 Z_out = [-1 ; Z];
 U_out = M*Z_out;
 Y = logistic(U_out);
 %Calculate the error
 E = testVal(k,:)' - Y;
 epochTestError(k) = E'*E;
end
classificationError = mean(epochTestError)                      