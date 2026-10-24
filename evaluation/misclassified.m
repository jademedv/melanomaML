% 1: Load the test features + final models
load('data/test_features.mat');
load('models/trained/final_rbfSVM.mat');
load('models/trained/final_knn.mat');
test_labels = test_features.Label;
feature_names = {'MeanR', 'MeanG', 'MeanB', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'};

% 2: Load the same test images as clean evaluation
% same test images as the clean evaluation (duplicates excluded)
test_path = '/Users/jade/Library/CloudStorage/OneDrive-UTS/melanoma_cancer_dataset/test';
imds_test = imageDatastore(test_path, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});
dupes = readtable('results/duplicate_check.csv'); % duplicates excluded
[~, test_base, test_ext] = fileparts(string(imds_test.Files));
keep = ~ismember(test_base + test_ext, string(dupes.TestImage));
imds_test = subset(imds_test, find(keep));
imgs = readall(imds_test);
if numel(imgs) ~= numel(test_labels)
    error("Image count does not match test_features");
end

% 3: Predictions + error groups
X = test_features(:, feature_names);
pred_svm = predict(rbfSVM, X);
pred_knn = predict(kNN, X);
is_malignant = test_labels == 'malignant';
fn_svm = is_malignant & pred_svm == 'benign'; % false negatives
fn_knn = is_malignant & pred_knn == 'benign';
fp_svm = ~is_malignant & pred_svm == 'malignant'; % false positives
fp_knn = ~is_malignant & pred_knn == 'malignant';
fn_both = fn_svm & fn_knn; % errors made by both models
fp_both = fp_svm & fp_knn;
fprintf("False negatives: SVM %d, k-NN %d, both %d\n", sum(fn_svm), sum(fn_knn), sum(fn_both));
fprintf("False positives: SVM %d, k-NN %d, both %d\n", sum(fp_svm), sum(fp_knn), sum(fp_both));

% 4: Image grid of a random sample of errors made by both models
rng(1);
n_show = 12;
groups = {fn_both, fp_both};
group_names = ["False negatives (malignant predicted benign)", ...
    "False positives (benign predicted malignant)"];
files = ["gallery_false_negatives.png", "gallery_false_positives.png"];
for g = 1:2
    idx = find(groups{g});
    idx = idx(randperm(numel(idx), min(n_show, numel(idx))));
    fig = figure('Color', 'w', 'Position', [100 100 1000 780]);
    tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, group_names(g) + ", both models", 'FontWeight', 'bold');
    for k = 1:numel(idx)
        nexttile;
        imshow(imgs{idx(k)});
        % title: test set index and the image's mean R, G, B
        title(sprintf('#%d  R%.0f G%.0f B%.0f', idx(k), ...
            test_features.MeanR(idx(k)), test_features.MeanG(idx(k)), test_features.MeanB(idx(k))), ...
            'FontSize', 8);
    end
    exportgraphics(fig, "results/figures/" + files(g), 'Resolution', 300);
end

% 5: Mean feature values for each outcome
% checks if the errors differ from correct predictions on the features
outcome_svm = strings(height(X), 1);
outcome_svm(is_malignant & pred_svm == 'malignant') = "TP";
outcome_svm(is_malignant & pred_svm == 'benign') = "FN";
outcome_svm(~is_malignant & pred_svm == 'malignant') = "FP";
outcome_svm(~is_malignant & pred_svm == 'benign') = "TN";
outcome_knn = strings(height(X), 1);
outcome_knn(is_malignant & pred_knn == 'malignant') = "TP";
outcome_knn(is_malignant & pred_knn == 'benign') = "FN";
outcome_knn(~is_malignant & pred_knn == 'malignant') = "FP";
outcome_knn(~is_malignant & pred_knn == 'benign') = "TN";
T = X;
T.Outcome = outcome_svm;
summary_svm = groupsummary(T, 'Outcome', 'mean', feature_names);
T.Outcome = outcome_knn;
summary_knn = groupsummary(T, 'Outcome', 'mean', feature_names);
disp(summary_svm);
disp(summary_knn);
writetable(summary_svm, 'results/error_feature_means_svm.csv');
writetable(summary_knn, 'results/error_feature_means_knn.csv');