% 1: Load data
addpath('scripts/'); % adds folder containing calculate_metrics function
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
% store groups together so script can loop through them
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

% 3: Cross validation setup
rng(1); % same seed + folds as tuning for consistency
cvp = cvpartition(response, 'KFold', 5);
classOrder = categorical({'benign'; 'malignant'});

% 4: Run SVM on each feature group
% hyperparameters are held at the tuned values
results = table();
for i = 1:numel(feature_sets)
    predictors = feature_sets{i};
    fprintf("Running RBF SVM with %s features...\n", feature_names{i});
    model = fitcsvm(predictors, response, ...
        'KernelFunction', 'gaussian', ...
        'KernelScale', 7, ...
        'BoxConstraint', 150, ...
        'Standardize', true, ...
        'ClassNames', classOrder, ...
        'CVPartition', cvp);
    % out of fold predictions + malignant class scores
    [predicted, score] = kfoldPredict(model);
    metrics = calculate_metrics(response, predicted, score(:, 2));
    metrics.FeatureSet = string(feature_names{i});
    metrics.Model = "RBF SVM";
    results = [results; metrics];
    fprintf("Finished %s features\n", feature_names{i});
end

% 5: Save results
% put identifying columns first
results = movevars(results, {'Model', 'FeatureSet'}, 'Before', 1);
disp(results);
writetable(results, 'results/feature_comparison_rbf.csv');
disp("Feature comparison results saved");