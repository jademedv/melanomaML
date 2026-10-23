% 1: Load training + test features
load('data/train_features.mat');
load('data/test_features.mat');
train_predictors = train_features(:, 1:end-1);
train_response = train_features.Label;
test_predictors = test_features(:, 1:end-1);
test_labels = test_features.Label;
class = categorical({'benign'; 'malignant'}); % class order: benign = neg, malignant = pos

% 2: Fit final models on full training set
% hyperparameters are the best configurations from tuning
rbfSVM = fitcsvm(train_predictors, train_response, ...
    'KernelFunction', 'gaussian', ...
    'KernelScale', 7, ...
    'BoxConstraint', 150, ...
    'Standardize', true, ...
    'ClassNames', class);
kNN = fitcknn(train_predictors, train_response, ...
    'NumNeighbors', 70, ...
    'Distance', 'cityblock', ...
    'DistanceWeight', 'squaredinverse', ...
    'Standardize', true, ...
    'ClassNames', class);
save('models/trained/final_rbfSVM.mat', 'rbfSVM');
save('models/trained/final_knn.mat', 'kNN');

% 3: Predict on test set
[prediction_rbf, score_rbf] = predict(rbfSVM, test_predictors);
[prediction_knn, score_knn] = predict(kNN, test_predictors);
model_names = ["RBF SVM"; "k-NN"];
preds = {prediction_rbf, prediction_knn};
scores = {score_rbf, score_knn};
num_models = numel(model_names);

% 4: Test set metrics + confusion matrix for both models
accuracy = zeros(num_models, 1);
recall = zeros(num_models, 1);
recall_lo = zeros(num_models, 1);
recall_hi = zeros(num_models, 1);
precision = zeros(num_models, 1);
specificity = zeros(num_models, 1);
f1 = zeros(num_models, 1);
auc = zeros(num_models, 1);
fpr = cell(num_models, 1); % false positive rate (falsely classified malignant)
tpr = cell(num_models, 1); % true positive rate (correctly classified malignant)
for m = 1:num_models
    C = confusionmat(test_labels, preds{m}, 'Order', class);
    tn = C(1, 1); 
    fp = C(1, 2);
    fn = C(2, 1); 
    tp = C(2, 2);
    accuracy(m) = (tp + tn) / sum(C, 'all');
    [recall(m), ci] = binofit(tp, tp + fn);
    recall_lo(m) = ci(1);
    recall_hi(m) = ci(2);
    precision(m) = tp / (tp + fp);
    specificity(m) = tn / (tn + fp);
    f1(m) = 2 * precision(m) * recall(m) / (precision(m) + recall(m));
    % ROC curve + AUC from malignant class score
    [fpr{m}, tpr{m}, ~, auc(m)] = perfcurve(test_labels, scores{m}(:, 2), 'malignant');
    % confusion matrix figure
    figure;
    confusionchart(C, {'benign', 'malignant'});
    title(model_names(m) + " test confusion matrix");
    saveas(gcf, "results/figures/confusion_" + strrep(model_names(m), " ", "_") + ".png");
end

% 5: Collect + save results
results_test = table(model_names, accuracy, recall, recall_lo, recall_hi, precision, specificity, f1, auc);
disp(results_test);
writetable(results_test, 'results/test_results.csv');

% 6: ROC curves for both models on one plot
figure;
hold on;
for m = 1:num_models
    plot(fpr{m}, tpr{m}, 'LineWidth', 1);
end
plot([0 1], [0 1], '--k'); % chance line
hold off;
xlabel('False positive rate');
ylabel('True positive rate');
legend(model_names + " (AUC " + string(round(auc, 3)) + ")", 'Location', 'southeast');
title('ROC curve (test set)');
saveas(gcf, 'results/figures/roc_test.png');

% 7: McNemar type paired test
[~, p_value] = testcholdout(prediction_rbf, prediction_knn, test_labels);
fprintf("RBF SVM vs. k-NN paired test; p-value: %.3f\n", p_value);

% 8: Recall + precision with 95% confidence intervals (figure only)
metrics = {'Recall', 'Precision'};
point = zeros(num_models, 2);
lo = zeros(num_models, 2);
hi = zeros(num_models, 2);
for m = 1:num_models
    C = confusionmat(test_labels, preds{m}, 'Order', class); 
    fp = C(1, 2);
    fn = C(2, 1); 
    tp = C(2, 2);
    [point(m, 1), ci] = binofit(tp, tp + fn); % recall interval
    lo(m, 1) = ci(1);
    hi(m, 1) = ci(2);
    [point(m, 2), ci] = binofit(tp, tp + fp); % precision interval
    lo(m, 2) = ci(1);
    hi(m, 2) = ci(2);
end
figure;
hold on;
x = [1, 2];
offset = [-0.08 0.08]; % offset so two models don't overlap
for m = 1:num_models
    errorbar(x + offset(m), point(m, :), point(m, :) - lo(m, :), hi(m, :) - point(m, :), 'o', 'LineWidth', 1);
end
hold off;
xlim([0.5 2.5]);
xticks(x);
xticklabels(metrics);
ylabel('Score');
legend(model_names, 'Location', 'southeast');
title('Test set recall and precision with 95% CI');
saveas(gcf, 'results/figures/test_ci.png');