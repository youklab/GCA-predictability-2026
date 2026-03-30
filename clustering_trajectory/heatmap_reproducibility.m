% heatmap_reproducibility.m
%
% Generate a multi-panel heatmap showing the reproducibility of
% trajectory-level vortex-core-size patterns across independent data sets.
%
% For each chunk_#.mat file, the script:
%   1. loads the total vortex-core-size trajectories,
%   2. standardizes missing entries,
%   3. orders trajectories by nearest-neighbor similarity in trajectory space,
%   4. interpolates each trajectory onto a common logarithmic time grid, and
%   5. displays the resulting heat map in a tiled figure.
%
% The figure uses chunks 1--19 so that every panel contains the same number
% of trajectories without augmentation.

clear;
close all;

%% Select the folder containing chunk_#.mat files
chunk_folder = uigetdir(pwd, 'Select the folder containing chunk_#.mat files');
if isequal(chunk_folder, 0)
    error('No folder selected.');
end

%% Analysis parameters
chunk_ids = 1:19;              % Independent trajectory sets shown in the figure.
num_sets = numel(chunk_ids);
cap_threshold = 12;            % Upper limit for displayed vortex-core size.
num_log_columns = 900;         % Resolution of the common logarithmic time grid.
target_num_rows = 2000;        % Number of trajectories displayed per panel.

%% Create output folder
output_folder = fullfile(chunk_folder, 'heatmap_reproducibility_results');
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% Create tiled figure
fig = figure('Color', 'w', 'Position', [50 50 1400 1600]);
layout = tiledlayout(5, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
axes_handles = gobjects(num_sets, 1);

for set_index = 1:num_sets
    chunk_id = chunk_ids(set_index);
    fprintf('Processing chunk %d (%d of %d)\n', chunk_id, set_index, num_sets);

    % Load one trajectory set.
    chunk_data = load(fullfile(chunk_folder, sprintf('chunk_%d.mat', chunk_id)), ...
        'subset_core_size');
    core_size = chunk_data.subset_core_size;

    if size(core_size, 1) ~= target_num_rows
        error('Chunk %d does not contain %d trajectories.', ...
            chunk_id, target_num_rows);
    end

    % Match the preprocessing used in the main analysis.
    core_size(isnan(core_size)) = -1;
    [num_rows, num_timepoints] = size(core_size);

    % Order rows by nearest-neighbor similarity in trajectory space.
    row_order = compute_similarity_order(core_size);
    core_size = core_size(row_order, :);

    % Cap displayed values to emphasize the low-value range.
    core_size(core_size > cap_threshold) = cap_threshold;

    % Interpolate onto a common logarithmic time grid.
    linear_time = 1:num_timepoints;
    log_time = logspace(0, log10(num_timepoints), num_log_columns);
    core_size_log = zeros(num_rows, num_log_columns);

    for row_idx = 1:num_rows
        core_size_log(row_idx, :) = interp1( ...
            linear_time, core_size(row_idx, :), log_time, 'linear', 'extrap');
    end

    % Plot one panel.
    ax = nexttile;
    imagesc(log_time, 1:num_rows, core_size_log);
    set(ax, 'XScale', 'log', 'FontSize', 8);
    caxis([0 cap_threshold]);
    colormap(ax, hot);
    title(sprintf('Set %d', chunk_id), 'FontSize', 9, 'FontWeight', 'normal');

    if set_index <= 16
        ax.XTickLabel = [];
    else
        ax.XTick = [1 100 10000];
        ax.XTickLabel = {'1', '100', '10000'};
    end

    if mod(set_index - 1, 4) ~= 0
        ax.YTickLabel = [];
    else
        ax.YTick = [1 target_num_rows];
        ax.YTickLabel = {'1', num2str(target_num_rows)};
    end

    axes_handles(set_index) = ax;
end

% Leave the final tile empty.
blank_ax = nexttile;
axis(blank_ax, 'off');

xlabel(layout, 'Backward time step, \tau', 'FontSize', 11);
ylabel(layout, 'Reordered trajectory index', 'FontSize', 11);

cb = colorbar(axes_handles(end), 'eastoutside');
cb.Layout.Tile = 'east';
ylabel(cb, sprintf('Total vortex core size (capped at %d)', cap_threshold));

output_file = fullfile(output_folder, 'heatmap_reproducibility.pdf');
print(fig, output_file, '-dpdf', '-painters');

fprintf('Saved heatmap figure to:\n%s\n', output_file);


function row_order = compute_similarity_order(core_size)
%COMPUTE_SIMILARITY_ORDER Order rows by nearest-neighbor similarity.
%   The first row is chosen as the trajectory whose last positive entry
%   occurs latest in backward time. Subsequent rows are appended greedily:
%   at each step, the next row is the remaining trajectory with the
%   smallest Euclidean distance to the most recently selected row.

    [~, num_timepoints] = size(core_size);
    last_positive_index = max((core_size > 0) .* (1:num_timepoints), [], 2);
    [~, first_row_index] = max(last_positive_index);

    remaining_indices = setdiff(1:size(core_size,1), first_row_index);
    row_order = first_row_index;

    while ~isempty(remaining_indices)
        last_selected = row_order(end);
        distances = vecnorm( ...
            core_size(remaining_indices, :) - core_size(last_selected, :), 2, 2);
        [~, min_index] = min(distances);
        row_order(end + 1) = remaining_indices(min_index); %#ok<AGROW>
        remaining_indices(min_index) = [];
    end
end
