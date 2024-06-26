% Open the file for reading
fid = fopen('output.txt', 'r');

% Read the data from the file into a vector
data_ = fscanf(fid, '%f'); 

% Close the file
fclose(fid);

% Split the data into two sets
n = length(data_); 
data_1 = data_(1:n);

x1 = 1:length(data_1);

% Open the file for reading
fid = fopen('loss.txt', 'r');

% Read the data from the file into a vector
data_ = fscanf(fid, '%f'); 

% Close the file
fclose(fid);

% Split the data into two sets
n = length(data_); 
data_2 = data_(1:n);

x2 = 1:length(data_2);
 
% Create a plot of the two sets of data
hold on;
plot(x1, data_1, x2, data_2); 

hold off;
xlabel('x');
ylabel('y');
grid on
legend('Logisstic Regression', 'MLP NN');