% Open the file for reading
fid = fopen('nn_graffed.txt', 'r');

% Read the data from the file into a vector
data = fscanf(fid, '%f');

% Close the file
fclose(fid);

% Split the data into two sets
n = length(data);
data1 = data(1:n/2);
data2 = data(n/2+1:end);

% Create x vectors for each set of data
x1 = 1:length(data1);
x2 = 1:length(data2);

% Create a plot of the two sets of data
hold on;
plot(x1, data1);
plot(x2, data2);
hold off;
xlabel('x');
ylabel('y');
title('Data Plot');
legend('Train Accuracy', 'Test Accuracy');
 