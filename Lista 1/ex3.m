clear;
clc;

data = load('ex1data3.txt');

training = data(1:30,:);
test = data(31:47,:);

Xt = training(:,1:5);
Xt = [ones(30,1) Xt];
Yt = training(:,6);

X = test(:,1:5);
X = [ones(17,1) X];
Y = test(:,6);

range = 5;
lambda = 0:range;

EQM_training = zeros(range,1);
EQM_test = zeros(range,1);

for i=1:(range+1)
  Ilambda = lambda(i) * eye(6);
  Ilambda(1,1) = 0;
  w = inv(Xt'*Xt + Ilambda)*Xt'*Yt;
  w
  #Treino
  Yt_calc = Xt*w;
  error = (Yt - Yt_calc)'*(Yt - Yt_calc);
  EQM_training(i) = sum(error)/30;
  
  #Teste
  Y_calc = X*w;
  error = (Y - Y_calc)'*(Y - Y_calc);
  EQM_test(i) = sum(error)/17;
end

figure(1);
plot(lambda, EQM_training, 'rx');
hold on;
xlabel('Lambda');
ylabel('EQM');

figure(2);
plot(lambda, EQM_test, 'bx');
hold on;
xlabel('Lambda');
ylabel('EQM');    
#X'[6x30] * X[30x6] +  I[6x6] = C[6x6]
#C[6x6] * X'[6x30] = D[6x30]
#D[6x30]*Y[30x1] = E[6X1]
#X[30x6]*w[6x1] = Y[30x1]
#Y'[1x30]*Y[30x1] = errorquad[1x1];