function out = columnsToRange0_1(data) 
%Receives a vector and standardizes all of its columns to the [0,1] range
out = zeros(rows(data), columns(data));
n = columns(data);

for i = 1:n
  out(:,i) = data(:,i)/max(data(:,i));
end