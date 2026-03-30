% graph_clustering_reproducibility.m
%
% Quantify reproducibility of graph-based trajectory clustering across
% independent trajectory sets. The script computes, for each chunk and for
% each backward-time window tau:
%   1) the number of connected components,
%   2) the fraction of trajectories that belong to singleton components, and
%   3) the size of the largest connected component.
%
% The script uses chunk_1.mat through chunk_19.mat. Each chunk is analyzed
% independently using the same distance threshold and graph-construction
% procedure used in the main trajectory-clustering analysis.

clear;
close all;

%% Select directory containing chunk_#.mat files

chunk_folder = uigetdir(pwd, 'Select folder containing chunk_#.mat files');
if isequal(chunk_folder, 0)
    error('No folder selected.');
end

chunk_ids = 1:19;
num_sets = numel(chunk_ids);

%% Analysis parameters

per_timestep_difference = 1;
num_sampled_windows = 200;
first_window_size = 20;

% Use chunk 1 only to determine the number of available timepoints.
reference_data = load(fullfile(chunk_folder, 'chunk_1.mat'), 'subset_core_size');
if ~isfield(reference_data, 'subset_core_size')
    error('chunk_1.mat does not contain subset_core_size.');
end
num_timepoints = size(reference_data.subset_core_size, 2);

window_sizes = floor(logspace(log10(first_window_size), ...
                              log10(num_timepoints), ...
                              num_sampled_windows));
window_sizes = unique(window_sizes, 'stable');
num_sampled_windows = numel(window_sizes);

%% Preallocate summary arrays

num_components = zeros(num_sets, num_sampled_windows);
singleton_fraction = zeros(num_sets, num_sampled_windows);
largest_component_size = zeros(num_sets, num_sampled_windows);

%% Analyze each chunk independently

for s = 1:num_sets

    chunk_id = chunk_ids(s);
    fprintf('\n=== Processing chunk %d (%d of %d) ===\n', chunk_id, s, num_sets);

    data = load(fullfile(chunk_folder, sprintf('chunk_%d.mat', chunk_id)), ...
                'subset_core_size');

    if ~isfield(data, 'subset_core_size')
        error('chunk_%d.mat does not contain subset_core_size.', chunk_id);
    end

    vortex_size = data.subset_core_size;
    vortex_size(isnan(vortex_size)) = -100;

    num_trajectories = size(vortex_size, 1);

    for j = 1:num_sampled_windows

        tau = window_sizes(j);
        trajectory_window = vortex_size(:, 1:tau);

        pairwise_distances = pdist(trajectory_window, 'euclidean');
        distance_matrix = squareform(pairwise_distances);

        distance_threshold = sqrt((per_timestep_difference ^ 2) * tau);

        adjacency_matrix = distance_matrix <= distance_threshold;
        adjacency_matrix(1:size(adjacency_matrix, 1) + 1:end) = 0;

        trajectory_graph = graph(adjacency_matrix);
        component_ids = conncomp(trajectory_graph, 'Type', 'weak');
        component_sizes = accumarray(component_ids(:), 1);

        num_components(s, j) = numel(component_sizes);
        singleton_fraction(s, j) = sum(component_sizes == 1) / num_trajectories;
        largest_component_size(s, j) = max(component_sizes);

        fprintf(['chunk %d | [%d/%d] tau=%d | components=%d | ' ...
                 'singleton fraction=%.3f | largest component=%d\n'], ...
                chunk_id, j, num_sampled_windows, tau, ...
                num_components(s, j), singleton_fraction(s, j), ...
                largest_component_size(s, j));
    end
end

%% Save summary data

output_folder = fullfile(chunk_folder, 'results_graph_clustering_reproducibility');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

save(fullfile(output_folder, 'graph_clustering_reproducibility.mat'), ...
    'window_sizes', 'chunk_ids', 'num_components', ...
    'singleton_fraction', 'largest_component_size');

%% Write long-format summary table

chunk_column = [];
tau_column = [];
num_components_column = [];
singleton_fraction_column = [];
largest_component_column = [];

for s = 1:num_sets
    n_tau = numel(window_sizes);
    chunk_column = [chunk_column; repmat(chunk_ids(s), n_tau, 1)]; %#ok<AGROW>
    tau_column = [tau_column; window_sizes(:)]; %#ok<AGROW>
    num_components_column = [num_components_column; num_components(s, :)']; %#ok<AGROW>
    singleton_fraction_column = [singleton_fraction_column; singleton_fraction(s, :)']; %#ok<AGROW>
    largest_component_column = [largest_component_column; largest_component_size(s, :)']; %#ok<AGROW>
end

summary_table = table(chunk_column, tau_column, num_components_column, ...
    singleton_fraction_column, largest_component_column, ...
    'VariableNames', {'chunk_id', 'tau', 'num_connected_components', ...
                      'singleton_fraction', 'largest_connected_component_size'});

writetable(summary_table, ...
    fullfile(output_folder, 'graph_clustering_reproducibility.xlsx'));

%% Compute means across chunks for plotting

mean_num_components = mean(num_components, 1);
mean_singleton_fraction = mean(singleton_fraction, 1);
mean_largest_component = mean(largest_component_size, 1);

%% Generate three-panel reproducibility figure

figure('Color', 'w', 'Position', [100 100 1200 350]);

mean_color = [0.55 0 0];
individual_color = [0.7 0.7 0.7];

% Panel (a): number of connected components
subplot(1, 3, 1);
hold on;
for s = 1:num_sets
    loglog(window_sizes, num_components(s, :), '-', ...
           'Color', individual_color, 'LineWidth', 0.8);
end
loglog(window_sizes, mean_num_components, '-', ...
       'Color', mean_color, 'LineWidth', 2);
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('\tau');
ylabel('Number of connected components');
title('(a)');
box off;

% Panel (b): fraction of trajectories in singleton components
subplot(1, 3, 2);
hold on;
for s = 1:num_sets
    loglog(window_sizes, singleton_fraction(s, :), '-', ...
           'Color', individual_color, 'LineWidth', 0.8);
end
loglog(window_sizes, mean_singleton_fraction, '-', ...
       'Color', mean_color, 'LineWidth', 2);
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('\tau');
ylabel('Fraction of trajectories in singleton components');
title('(b)');
ylim([1e-2 1]);
yticks([1e-2 1e-1 1]);
box off;

% Panel (c): size of the largest connected component
subplot(1, 3, 3);
hold on;
for s = 1:num_sets
    loglog(window_sizes, largest_component_size(s, :), '-', ...
           'Color', individual_color, 'LineWidth', 0.8);
end
loglog(window_sizes, mean_largest_component, '-', ...
       'Color', mean_color, 'LineWidth', 2);
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('\tau');
ylabel('Largest connected-component size');
title('(c)');
box off;

print(gcf, fullfile(output_folder, 'graph_clustering_reproducibility.pdf'), ...
      '-dpdf', '-painters');

fprintf('\nSaved outputs to:\n%s\n', output_folder);
