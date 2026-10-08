load('data/train_features.mat');

predictors = train_features(:, 1:end-1);
response = train_features.Label;
class = categorical({'benign'; 'malignant'});

rng(1);
cvp = cvpartition(response, 'KFold', 5);

kernel = [0.1 0.3 0.66 1 2 5 10];
box = [0.1 1 10 100];
num_ks = numel(kernel);
num_bc = numel(box);
num_runs = num_ks * num_bc;

ks_col = zeros(num_runs, 1);
bc_col = zeros(num_runs, 1);
accuracy = zeros(num_runs, 1);
recall = zeros(num_runs, 1);
precision = zeros(num_runs, 1);
specificity = zeros(num_runs, 1);
f1 = zeros(num_runs, 1);
recall_grid = zeros(num_ks, num_bc);

idx = 0;
for i = 1:num_ks
    for j = 1:num_bc
        idx = idx + 1;
        fprintf("Run %d / %d: kernel scale %.2f, box constraint %.1f\n", ...
            idx, num_runs, kernel(i), box(j));

        model = fitcsvm(predictors, response, ...
            'KernelFunction', 'gaussian', ...
            'KernelScale', kernel(i), ...
            'BoxConstraint', box(j), ...
            'Standardize', true, ...
            'ClassNames', class, ...
            'CVPartition', cvp);

        prediction = kfoldPredict(model);
        C = confusionmat(response, prediction, 'Order', class);
        tn = C(1,1);
        fp = C(1,2);
        fn = C(2,1);
        tp = C(2,2);

        ks_col(idx) = kernel(i);
        bc_col(idx) = box(j);
        accuracy(idx) = (tp + tn) / sum(C, 'all');
        recall(idx) = tp / (tp + fn);
        precision (idx) = tp / (tp + fp);
        specificity(idx) = tn / (tn + fp);
        f1(idx) = 2 * precision(idx) * recall(idx) / (precision(idx) + recall(idx));
        recall_grid(i, j) = recall(idx) * 100;
    end
end

results_rbf = table(ks_col, bc_col, accuracy, recall, precision, specificity, f1, ...
    'VariableNames', {'KernelScale', 'BoxConstraint', 'Accuracy', ...
    'Recall', 'Precision', 'Specificity', 'F1'});
disp(results_rbf);

eval_rbf = results_rbf(results_rbf.Precision >= 0.80, :);
if isempty(eval_rbf)
    disp("No setting reached at least 80% precision.")
    eval_rbf = results_rbf;
end
[~, best_idx] = max(eval_rbf.Recall);
best_rbf = eval_rbf(best_idx, :);
disp("Best RBF SVM setting:");
disp(best_rbf);

% save results
writetable(results_rbf, 'results/tune_rbf.csv');

figure;
h = heatmap(string(box), string(kernel), recall_grid);
h.CellLabelFormat = '%.1f';
xlabel('Box Constraint');
ylabel('Kernel Scale');
title('RBF SVM malignant recall, 5-fold CV');
saveas(gcf, 'results/figures/rbf_recall_heatmap.png');