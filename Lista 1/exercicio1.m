clc;
clear;

data=load('ex1data1.txt');
x=data(:,1);
y=data(:,2);

figure(1);
plot(x, y, 'rx');

#Definindo de acordo com a questao
epochs=1000;
alpha=0.001;

w0=1;
w1=1;

eqm=zeros(epochs,1);

for i=1:epochs
  #Sao 97 valores, gera permutacao aleatoria dos valores
  perm = randperm(97);
  x = x(perm);
  y = y(perm);
  #Para cada par entrada,saida
  for j=1:97
    error=y(j)-x(j)*w1-w0;
    w0=w0+alpha*error;
    w1=w1+alpha*error*x(j);
    #Soma o erro quadratico
    eqm(i)=eqm(i)+error^2;
  end
  #No final divide para obter a media
  eqm(i)=eqm(i)/97;
end

figure(2);
plot(eqm(:,:));
hold on;
xlabel('Epochs');
ylabel('EQM');

figure(3);
plot(x, y, 'rx');
hold on;
fplot(@(x) w1*x+w0, [min(x), max(x)],'b');