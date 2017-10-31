close all;
clc;
clear;

%Load matrices
X = load("ex3data2.data");
%X(:,1:13) = columnsToRange0_1(X(:,1:13));
%X = X./100;
X = columnsToRange0_1(X);
%Randomize data, and separate in training, validation and test sets
[Xrow, Xcol] = size(X);

perm = randperm(Xrow);
X(perm,:);

trainAttr = X(1:306,1:13);
trainVal = X(1:306,14);
trainNum = rows(trainAttr);


valAttr = X(307:406,1:13);
valVal = X(307:406,14);
valNum = rows(valAttr);

testAttr = X(407:506,1:13);
testVal = X(407:506,14);
testNum = rows(testAttr);

%Defining variables
hiddenNeurs = 10; %Number of hidden neurons
outNeurs = 1; %Number of out neurons
alpha = 0.1;

trainError = [];
valError = [];
W = rand(hiddenNeurs,Xcol);
M = rand(outNeurs,hiddenNeurs+1);
firstEpoch = true;
%While the difference between the past validation error and the present
%validation error is less than the tolerance, do
%while true
for m=1:200
  %Train the weights on training set
  epochTrainingError = zeros(trainNum,1);
  for i=1:trainNum
    %Start with hidden layer
    X_hid = [-1 ; trainAttr(i,:)']; %Construct the input vector
    U_hid = W*X_hid; %Calculate the sum of weights+attributes
    Z = logistic(U_hid); %Apply logistic function
   %Then use it as input for the output layer, repeat the same process
   %with the difference of not using the activation function for the output
    Z_out = [-1 ; Z];
    U_out = M*Z_out;
    Y = U_out;
    %Calculate the error
    E = trainVal(i,:)' - Y;
    
    epochTrainingError(i) = E'*E;
    
    %Calculate the local gradients
    G_out = E; %Output gradient
    G_hid = (Z.*(1-Z)) .* (M(:,2:end)'*G_out); %Hidden layer gradient
    
    %Update weights
    M = M + alpha*G_out*Z_out'; %Output layer
    W = W + alpha*G_hid*X_hid'; %Hidden layer
  end
  %Save the avg of the training error for later
  trainError = [trainError; mean(epochTrainingError)];
  
  epochValError = zeros(valNum, 1);
  %Calculate the values on the validation set 
  for j=1:valNum
    %Start with hidden layer
    X_hid = [-1 ; valAttr(j,:)']; %Construct the input vector
    U_hid = W*X_hid; %Calculate the sum of weights+attributes
    Z = logistic(U_hid); %Apply logistic function
   %Then use it as input for the output layer, repeat the same process
   %with the difference of not using the activation function for the output
    Z_out = [-1 ; Z];
    U_out = M*Z_out;
    Y = U_out;
    %Calculate the error
    E = valVal(j,:)' - Y;
    epochValError(j) = E'*E;
  end
  valError = [valError; mean(epochValError)];
  
  if (!firstEpoch)
    %If the error grows, end the training
    if (valError(end) - valError(end-1) > 0)
      %break;
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

%Get classification error for test set,calculate the values on the test set
%and plot values
figure(2);
epochTestError = zeros(testNum, 1);
for k=1:testNum
 %Start with hidden layer
 X_hid = [-1 ; testAttr(k,:)']; %Construct the input vector
 U_hid = W*X_hid; %Calculate the sum of weights+attributes
 Z = logistic(U_hid); %Apply logistic function
 %Then use it as input for the output layer, repeat the same process
 %with the difference of not using the activation function for the output
 Z_out = [-1 ; Z];
 U_out = M*Z_out;
 Y = U_out;
 %Calculate the error
 E = testVal(k,:)' - Y;
 plot(testAttr(k,1), testVal(k,:), 'bo');
 hold on;
 plot(testAttr(k,1), Y, 'rx');
 hold on;
 epochTestError(k) = E^2;
end
classificationError = mean(epochTestError)
hold on;
xlabel("Attr 1");
ylabel("House Price");
h = legend ({"Actual Value"}, "Predicted Value");
legend (h, "location", "northeastoutside");                       