function [] = plot_learning_server_profile(j, name, csv_file)

data = csvread(csv_file, 3);
NUM_ITERS = size(data, 1);

figure(j);
clf;
bar(1:NUM_ITERS, data(:, 2:4)/60, 0.5, 'stack');
title(sprintf('Timings for %s', name));
xlabel('Episode');
ylabel('Duration (min)');
legend({'Dynamics', 'Policy', 'Save'}, 'Location', 'NorthWest');

end
