clc;
clear;

data = load('ex1data2.txt');
X = data(:,1:2)';
X = [ones(1,47); X];
Y = data(:,3);

function res = error(yi, W, Xi)
  res = yi - W'*Xi;
endfunction

epochs = 100;
alpha = 0.001;

W = [1; 1; 1];
eqm = zeros(epochs,1);

for i=1:epochs
  perm = randperm(47);
  X = X(:,perm);
  Y = Y(perm);
  for j=1:47
    error_val = error(Y(j), W, X(:,j));
    eqm(i) = eqm(i) + error_val^2;
    W = W + alpha * error_val * X(:,j);
  endfor
  eqm(i) = eqm(i) / 47;  
endfor

figure(1);
plot(eqm);
hold on;
xlabel('Epochs');
ylabel('EQM'); 