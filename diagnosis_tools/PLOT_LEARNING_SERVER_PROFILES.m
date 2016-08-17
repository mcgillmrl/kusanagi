% PLOT_LEARNING_SERVER_PROFILES.m

logs = {'inferno', 'gorgona', 'ithaca', 'moist', 'numenor', 'capilano'};
for j = 1:length(logs),
    name = logs{j};
    csvpath = sprintf('/media/diskstation/learn_to_swim/DELETE_ME/anqi_test_remotes/uturn_%s_timings.csv', name);
    pngpath = sprintf('/media/diskstation/learn_to_swim/DELETE_ME/anqi_test_remotes/uturn_%s_timings.png', name);
    plot_learning_server_profile(name, csvpath, j, pngpath);
end;
