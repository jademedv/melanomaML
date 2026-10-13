% Baseline cross-validation comparisons, not final tuned-model performance
load('../data/train_features.mat');

response = train_features.Label;

% Define feature groups
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

rng(1);
cvp = cvpartition(response, 'KFold', 5);

classOrder = categorical({'benign'; 'malignant'});

results = table();

for i = 1:numel(feature_sets)

    predictors = feature_sets{i};

    fprintf("Running k-NN with %s features...\n", feature_names{i});

    model = fitcknn(predictors, response, ...
        'Standardize', true, ...
        'ClassNames', classOrder, ...
        'CVPartition', cvp);

    predicted = kfoldPredict(model);

    metrics = calculate_metrics(response, predicted);

    metrics.FeatureSet = string(feature_names{i});
    metrics.Model = "k-NN";

    results = [results; metrics];

    fprintf("Finished %s features\n", feature_names{i});

end

results = movevars(results, {'Model', 'FeatureSet'}, 'Before', 1);

disp(results);

writetable(results, '../results/feature_comparison_knn.csv');

disp("k-NN feature comparison results saved");