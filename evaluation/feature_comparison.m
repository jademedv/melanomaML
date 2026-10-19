addpath('scripts/');
load('data/train_features.mat');

response = train_features.Label;

% Define feature groups
colour_features = train_features(:, ...
    {'MeanR', 'MeanG', 'MeanB'});

texture_features = train_features(:, ...
    {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

combined_features = train_features(:, ...
    {'MeanR', 'MeanG', 'MeanB', ...
    'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

% Store groups together so we can loop through them
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

% 5-fold cross-validation
rng(1);
cvp = cvpartition(response, 'KFold', 5);

% Class order: malignant is the positive class
classOrder = categorical({'benign'; 'malignant'});

% Store results
results = table();

for i = 1:numel(feature_sets)

    predictors = feature_sets{i};

    fprintf("Running RBF SVM with %s features...\n", feature_names{i});

    % RBF SVM
    model = fitcsvm(predictors, response, ...
        'KernelFunction', 'gaussian', ...
        'KernelScale', 7, ...
        'BoxConstraint', 150, ...
        'Standardize', true, ...
        'ClassNames', classOrder, ...
        'CVPartition', cvp);

    [predicted, score] = kfoldPredict(model);

    metrics = calculate_metrics(response, predicted, score(:, 2));

    metrics.FeatureSet = string(feature_names{i});
    metrics.Model = "RBF SVM";

    results = [results; metrics];
    fprintf("Finished %s features\n", feature_names{i});

end

% Put identifying columns first
results = movevars(results, {'Model', 'FeatureSet'}, 'Before', 1);

disp(results);

writetable(results, 'results/feature_comparison_rbf.csv');

disp("Feature comparison results saved");