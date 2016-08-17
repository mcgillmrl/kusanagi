% PLOT_LEARNING_SERVER_PROFILES.m

logs = {'inferno', 'gorgona', 'ithaca', 'moist', 'numenor', 'capilano'};
for j = 1:length(logs),
    name = logs{j};
    csv_file = sprintf('/media/diskstation/learn_to_swim/DELETE_ME/anqi_test_remotes/uturn_%s_timings.csv', name);
    plot_learning_server_profile(j, name, csv_file);
    print(gcf, '-dpng', sprintf('uturn_%s_timings.png', name));
end;