function out = columnsToRange0_1(data) 
%Receives a matrix and standardizes all of its columns to the [0,1] range
out = zeros(rows(data), columns(data));
n = columns(data);

for i = 1:n
  maxVal = max(data(:,i));
  if (maxVal < 0.0005)
    maxVal = 1;
  end
  out(:,i) = data(:,i)/maxVal;
end