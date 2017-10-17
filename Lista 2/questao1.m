close all;
clc;
clear;

data = load('ex2data1.txt');
%Normalizando as variaveis de entrada
data(:,1:2) = columnsToRange0_1(data(:,1:2));

train = data(1:70,:);
test = data(71:100,:);

alpha = 0.001;
epochs = 1000;

function out = logistic(x)
  out = 1/(1+exp(-x));
end

function out = category(x)
  out = round(x);
end

function out = miss(expected, actual)
  if (expected != actual)
    out = 1;
  else
    out = 0;
  end
end

%Chute inicial de coeficientes
W = [1; 1; 1];

%Dados de treino
X = [ones(1,70); train(:,1:2)'];
Y = train(:,3);

%Dados de teste
Xt = [ones(1,30); test(:,1:2)'];
Yt = test(:,3);

eqm = zeros(epochs,1);
wrong_samples = zeros(epochs,1);

for j=1:epochs
  %Permuta as amostras
  perm = randperm(70);
  X = X(:, perm);
  Y = Y(perm);
  %Coeficientes no conjunto de treinamento
  for i=1:70
    error = Y(i)-logistic(W'*X(:,i));
    W = W + alpha*error*X(:,i);
  end
  %Erro no conjunto de testes
  for k=1:30
    error_test = Yt(k)-logistic(W'*Xt(:,k));
    eqm(j) = eqm(j) + (error_test)^2;
    wrong_samples(j) += miss(Yt(k), category(logistic(W'*Xt(:,k))));
  end
  eqm(j) = eqm(j)/30;
end

figure(1);
hold on;
scatter(data(:,1), data(:,2), 3, data(:,3));

figure(2);
plot([-W(1)/W(2), 0], [0, -W(1)/W(3)], 'b');
hold on;
for i=1:30
  expected_c = test(i,3);
  actual_c = category(logistic(W'*Xt(:,i))); 
  if (expected_c == 0 && expected_c != actual_c)
    plot(Xt(2,i), Xt(3,i), 'gx'); 
  elseif (expected_c == 0 && expected_c == actual_c)
    plot(Xt(2,i), Xt(3,i), 'go');
  elseif (expected_c == 1 && expected_c != actual_c)
    plot(Xt(2,i), Xt(3,i), 'rx');
  elseif (expected_c == 1 && expected_c == actual_c)
    plot(Xt(2,i), Xt(3,i), 'ro');
  endif 
end

   
figure(3);
plot(eqm);
hold on;
xlabel('Epoch');
ylabel('Error');

figure(4);
plot(wrong_samples);
hold on;
xlabel('Epoch');
ylabel('# of points in wrong categories');  