function [] = plot_learning_server_profile(name, csvpath, fig_j, pngpath)

data = csvread(csvpath, 3);
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
legend({'Dynamics', 'Policy', 'Save', 'Wait till next'}, 'Location', 'NorthWest');
grid on;
ax = axis();
ax(1) = 0; ax(2) = ceil(ax(2)/5)*5; ax(3) = 0; ax(4) = ceil(ax(4)/5)*5;
axis(ax);

if nargin >= 4,
  print(h_fig, '-dpng', pngpath);
end

end
