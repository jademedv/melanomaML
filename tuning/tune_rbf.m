load('data/train_features.mat');

predictors = train_features(:, 1:end-1);
response = train_features.Label;
class = categorical({'benign'; 'malignant'});

seed = 1;
run_id = 1;
rng(seed);
cvp = cvpartition(response, 'KFold', 5);
run_num = "v" + run_id;

%kernel = [0.1 0.3 0.6 1 2 4 5 7 10];
%box = [0.1 1 10 50 100 150];
kernel = [3 4 5 6 7 8 10 12 15];
box = [10 50 100 150 200 300 500 1000];
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
auc = zeros(num_runs, 1);
recall_lo = zeros(num_runs, 1);
recall_hi = zeros(num_runs, 1);

recall_grid = zeros(num_ks, num_bc);
%precision_grid = zeros(num_ks, num_bc);
%f1_grid = zeros(num_ks, num_bc);

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

        [prediction, score] = kfoldPredict(model);
        C = confusionmat(response, prediction, 'Order', class);
        tn = C(1,1);
        fp = C(1,2);
        fn = C(2,1);
        tp = C(2,2);

        ks_col(idx) = kernel(i);
        bc_col(idx) = box(j);
        accuracy(idx) = (tp + tn) / sum(C, 'all');

        [recall(idx), ci] = binofit(tp, tp + fn);
        recall_lo(idx) = ci(1);
        recall_hi(idx) = ci(2);

        %recall(idx) = tp / (tp + fn);
        precision (idx) = tp / (tp + fp);
        specificity(idx) = tn / (tn + fp);

        f1(idx) = 2 * precision(idx) * recall(idx) / (precision(idx) + recall(idx));
        [~, ~, ~, auc(idx)] = perfcurve(response, score(:, 2), 'malignant');
        recall_grid(i, j) = recall(idx) * 100;
        %precision_grid(i, j) = precision(idx) * 100;
        %f1_grid(i, j) = f1(idx) * 100;
    end
end

results_rbf = table(ks_col, bc_col, accuracy, recall, recall_lo, recall_hi, precision, specificity, f1, auc, ...
    'VariableNames', {'KernelScale', 'BoxConstraint', 'Accuracy', ...
    'Recall', 'RecallLo', 'RecallHi', 'Precision', 'Specificity', 'F1', 'AUC'});
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
writetable(results_rbf, "results/tune_rbf_" + run_num + ".csv");

figure;
h = heatmap(string(box), string(kernel), recall_grid);
h.CellLabelFormat = '%.1f';
xlabel('Box Constraint');
ylabel('Kernel Scale');
title('RBF SVM malignant recall, 5-fold CV');
saveas(gcf, "results/figures/rbf_recall_heatmap_" + run_num + ".png");

figure;
scatter(results_rbf.Precision * 100, results_rbf.Recall * 100, 40, log10(results_rbf.KernelScale), 'filled');
hold on;
xline(80, '--k');
hold off;
colorbar;
xlabel('Precision (%)');
ylabel('Recall (%)');
title('RBF SVM recall vs precision');
saveas(gcf, "results/figures/rbf_recallprec_" + run_num + ".png");