% Open the file for reading
fid = fopen('cost_nn.txt', 'r');

% Read the data from the file into a vector
data_ = fscanf(fid, '%f'); 

% Close the file
fclose(fid);

% Split the data into two sets
n = length(data_); 
data__ = data_(1:n);

x1 = 1:length(data__);

% Create a plot of the two sets of data
hold on;
plot(x1, data__); 
hold off;
xlabel('x');
ylabel('y');