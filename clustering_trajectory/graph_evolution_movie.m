%% graph_evolution_movie.m
% Generate graph-evolution movies for trajectory similarity networks.
%
% For each chunk_*.mat file in a user-selected folder, this script builds a
% sequence of trajectory-similarity graphs across logarithmically sampled
% backward-time windows and exports the evolution as GIF, AVI, and MP4.
%
% Required variables in each chunk file:
%   - subset_core_size
%   - subset_final_pattern_type
%
% The graph is defined by connecting two trajectories when the Euclidean
% distance between their total-vortex-core-size time series is less than or
% equal to sqrt(tau), where tau is the backward-time window size.

clear;
close all;

%% Analysis parameters
outputFolderName = 'graph_evolution_movies';
numSampledWindows = 200;
startColumn = 20;
differencePerTimeStep = 1;

figurePosition = [100, 100, 1600, 1200];
frameRate = 4;
gifDelayTime = 0.25;
nodeMarkerSize = 4;
edgeAlpha = 0.4;
edgeColor = [0.3, 0.3, 0.3];
nodeColormap = [0.2, 0.6, 0.6; 0.9, 0.7, 0.0; 0.8, 0.2, 0.2];

%% Select input directory
inputFolder = uigetdir('', 'Select the folder containing chunk_*.mat files');
if isequal(inputFolder, 0)
    error('No folder selected.');
end

chunkFiles = dir(fullfile(inputFolder, 'chunk_*.mat'));
if isempty(chunkFiles)
    error('No chunk_*.mat files were found in the selected folder.');
end

outputFolder = fullfile(inputFolder, outputFolderName);
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

if isempty(gcp('nocreate'))
    parpool;
end

%% Process all chunk files in parallel
parfor fileIdx = 1:numel(chunkFiles)
    chunkName = chunkFiles(fileIdx).name;
    chunkPath = fullfile(inputFolder, chunkName);

    data = load(chunkPath, 'subset_core_size', 'subset_final_pattern_type');

    if ~isfield(data, 'subset_core_size') || ~isfield(data, 'subset_final_pattern_type')
        error('File %s does not contain the required variables.', chunkName);
    end

    coreSize = data.subset_core_size;
    finalPatternType = data.subset_final_pattern_type(:);

    coreSize(isnan(coreSize)) = -100;

    numTimeSteps = size(coreSize, 2);
    sampledWindows = floor(logspace(log10(startColumn), log10(numTimeSteps), numSampledWindows));
    sampledWindows = unique(sampledWindows, 'stable');

    createGraphEvolutionMovie(
        coreSize, ...
        finalPatternType, ...
        sampledWindows, ...
        differencePerTimeStep, ...
        outputFolder, ...
        chunkName, ...
        figurePosition, ...
        frameRate, ...
        gifDelayTime, ...
        nodeMarkerSize, ...
        edgeAlpha, ...
        edgeColor, ...
        nodeColormap);

    fprintf('Finished movie generation for %s\n', chunkName);
end

fprintf('Completed movie generation for all chunk files.\n');

%% Local function
function createGraphEvolutionMovie(coreSize, finalPatternType, sampledWindows, ...
    differencePerTimeStep, outputFolder, chunkName, figurePosition, ...
    frameRate, gifDelayTime, nodeMarkerSize, edgeAlpha, edgeColor, nodeColormap)

[~, baseName, ~] = fileparts(chunkName);

gifPath = fullfile(outputFolder, sprintf('%s.gif', baseName));
aviPath = fullfile(outputFolder, sprintf('%s.avi', baseName));
mp4Path = fullfile(outputFolder, sprintf('%s.mp4', baseName));

movieFigure = figure('Position', figurePosition, 'Color', 'w');

aviWriter = VideoWriter(aviPath);
aviWriter.FrameRate = frameRate;
open(aviWriter);

mp4Writer = VideoWriter(mp4Path, 'MPEG-4');
mp4Writer.FrameRate = frameRate;
open(mp4Writer);

gifInitialized = false;
windowOrder = fliplr(1:numel(sampledWindows));

for frameIdx = 1:numel(windowOrder)
    windowIdx = windowOrder(frameIdx);
    tau = sampledWindows(windowIdx);

    subsetData = coreSize(:, 1:tau);
    distanceVector = pdist(subsetData, 'euclidean');
    distanceMatrix = squareform(distanceVector);

    distanceThreshold = sqrt(differencePerTimeStep^2 * tau);
    adjacencyMatrix = distanceMatrix <= distanceThreshold;
    adjacencyMatrix(1:size(adjacencyMatrix, 1) + 1:end) = 0;

    similarityGraph = graph(adjacencyMatrix);

    clf(movieFigure);
    set(movieFigure, 'Position', figurePosition);
    plot(similarityGraph, ...
        'Layout', 'force', ...
        'NodeCData', finalPatternType, ...
        'MarkerSize', nodeMarkerSize, ...
        'EdgeAlpha', edgeAlpha, ...
        'EdgeColor', edgeColor);
    colormap(nodeColormap);
    caxis([1, 3]);
    colorbar off;
    title(sprintf('Backward-time window: [1, %d]', tau), 'FontSize', 12);
    drawnow;

    frame = getframe(movieFigure);
    writeVideo(aviWriter, frame);
    writeVideo(mp4Writer, frame);

    imageFrame = frame2im(frame);
    [indexedFrame, colorMap] = rgb2ind(imageFrame, 256);
    if ~gifInitialized
        imwrite(indexedFrame, colorMap, gifPath, 'gif', ...
            'LoopCount', Inf, 'DelayTime', gifDelayTime);
        gifInitialized = true;
    else
        imwrite(indexedFrame, colorMap, gifPath, 'gif', ...
            'WriteMode', 'append', 'DelayTime', gifDelayTime);
    end

    fprintf('Processed frame %d of %d for %s (tau = %d)\n', ...
        frameIdx, numel(windowOrder), chunkName, tau);
end

close(aviWriter);
close(mp4Writer);
close(movieFigure);
end
