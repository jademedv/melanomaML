% 1: Load data
addpath('scripts/'); % calculate_metrics folder
load('data/train_features.mat');
response = train_features.Label;

% 2: Define feature groups
colour_features = train_features(:, ...
    {'MeanR', 'MeanG', 'MeanB'});
texture_features = train_features(:, ...
    {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
combined_features = train_features(:, ...
    {'MeanR', 'MeanG', 'MeanB', ...
    'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
feature_sets = {
    colour_features;
    texture_features;
    combined_features
    };
feature_names = {
    'Colour';
    'Texture';
    'Combined'
    };

% 3: Cross validation set up (same seed + folds)
rng(1);
cvp = cvpartition(response, 'KFold', 5);
classOrder = categorical({'benign'; 'malignant'});

% 4: Run k-NN on each feature group
results = table();
for i = 1:numel(feature_sets)
    predictors = feature_sets{i};
    fprintf("Running k-NN with %s features...\n", feature_names{i});
    model = fitcknn(predictors, response, ...
        'NumNeighbors', 70, ...
        'Distance', 'cityblock', ...
        'DistanceWeight', 'squaredinverse', ...
        'Standardize', true, ...
        'ClassNames', classOrder, ...
        'CVPartition', cvp);
    % out of fold predictions + malignant class scores
    [predicted, score] = kfoldPredict(model);
    metrics = calculate_metrics(response, predicted, score(:, 2));
    metrics.FeatureSet = string(feature_names{i});
    metrics.Model = "k-NN";
    results = [results; metrics];
    fprintf("Finished %s features\n", feature_names{i});
end

% 5: Save results (identifying columsn first)
results = movevars(results, {'Model', 'FeatureSet'}, 'Before', 1);
disp(results);
writetable(results, 'results/feature_comparison_knn.csv');
disp("k-NN feature comparison results saved");