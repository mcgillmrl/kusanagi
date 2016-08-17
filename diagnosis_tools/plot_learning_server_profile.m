function [] = plot_learning_server_profile(name, csvpath, fig_j, pngpath)

data = csvread(csv_file, 3);
NUM_ITERS = size(data, 1);

if nargin < 3,
  h_fig = figure();
else
  h_fig = figure(fig_j);
end

clf;
bar(1:NUM_ITERS, data(:, 2:5)/60, 0.5, 'stack');
title(sprintf('Timings for %s', name));
xlabel('Episode');
ylabel('Duration (min)');
legend({'Dynamics', 'Policy', 'Save'}, 'Location', 'NorthWest');
grid on;

if nargin >= 4,
  print(h_fig, '-dpng', pngpath);
end

end
