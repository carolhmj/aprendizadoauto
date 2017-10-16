clc;
clear;
data = load('ex1data2.txt');
X = data(:,1:2);
X = [ones(47,1) X];
Y = data(:,3);
function res = error(yi, W, Xi)
  res = yi - Xi*W';
endfunction

epochs = 100;
alpha = 0.001;

W = [1 1 1];
eqm = zeros(epochs,1);

for i=1:epochs
  error_sum = zeros(1,3);
  
  for j=1:47
    eqm(i) = eqm(i) + error(Y(j), W, X(j,:))^2;
    error_sum = error_sum + error(Y(j), W, X(j,:))*X(j,:);
  endfor
  
  W = W + alpha * error_sum * X;
  eqm(i) = eqm(i) / 47;  
endfor
plot(eqm);
hold on;
xlabel('Epochs');
ylabel('EQM'); 