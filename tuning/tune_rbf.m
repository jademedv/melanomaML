% 1: Load data + set up cross validation
load('data/train_features.mat');
predictors = train_features(:, 1:end-1); % features
response = train_features.Label; % classes
class = categorical({'benign'; 'malignant'});
%increases for each new run so earlier results are not overwritten
seed = 1;
run_id = 1;
rng(seed); % fixed seeds so folds are reproducible
cvp = cvpartition(response, 'KFold', 5); % one shared stratified 5-fold partition
run_num = "v" + run_id;

% 2: Define hyperparameter grid
kernel = [3 4 5 6 7 8 10 12 15]; % kernel scales
box = [10 50 100 150 200 300 500 1000]; % box constraints
num_ks = numel(kernel);
num_bc = numel(box);
num_runs = num_ks * num_bc; % total num of configurations

% 3: Preallocate result arrays
ks_col = zeros(num_runs, 1);
bc_col = zeros(num_runs, 1);
accuracy = zeros(num_runs, 1);
recall = zeros(num_runs, 1);
precision = zeros(num_runs, 1);
specificity = zeros(num_runs, 1);
f1 = zeros(num_runs, 1);
auc = zeros(num_runs, 1);
recall_lo = zeros(num_runs, 1); % lower bound of recall CI
recall_hi = zeros(num_runs, 1); % upper bound of recall CI
recall_grid = zeros(num_ks, num_bc); % kernel x box for recall heatmap

% 4: Grid search
idx = 0;
for i = 1:num_ks
    for j = 1:num_bc
        idx = idx + 1;
        fprintf("Run %d / %d: kernel scale %.2f, box constraint %.1f\n", ...
            idx, num_runs, kernel(i), box(j));
        % train an RBF SVM on each fold; features are standardised within
        % each fold
        model = fitcsvm(predictors, response, ...
            'KernelFunction', 'gaussian', ...
            'KernelScale', kernel(i), ...
            'BoxConstraint', box(j), ...
            'Standardize', true, ...
            'ClassNames', class, ...
            'CVPartition', cvp);
        % out of fold predictions + scores for every training image
        [prediction, score] = kfoldPredict(model);
        % confusion matrix (rows = true class, columns = predicted)
        C = confusionmat(response, prediction, 'Order', class);
        tn = C(1,1);
        fp = C(1,2);
        fn = C(2,1);
        tp = C(2,2);
        %store config and its metrics
        ks_col(idx) = kernel(i);
        bc_col(idx) = box(j);
        accuracy(idx) = (tp + tn) / sum(C, 'all');
        [recall(idx), ci] = binofit(tp, tp + fn); % recall with 95% Clopper-Pearson interval
        recall_lo(idx) = ci(1);
        recall_hi(idx) = ci(2);
        precision (idx) = tp / (tp + fp);
        specificity(idx) = tn / (tn + fp);
        f1(idx) = 2 * precision(idx) * recall(idx) / (precision(idx) + recall(idx));
        [~, ~, ~, auc(idx)] = perfcurve(response, score(:, 2), 'malignant'); % AUC from malignant class
        recall_grid(i, j) = recall(idx) * 100;
    end
end

% 5: Collect results into a table
results_rbf = table(ks_col, bc_col, accuracy, recall, recall_lo, recall_hi, precision, specificity, f1, auc, ...
    'VariableNames', {'KernelScale', 'BoxConstraint', 'Accuracy', ...
    'Recall', 'RecallLo', 'RecallHi', 'Precision', 'Specificity', 'F1', 'AUC'});
disp(results_rbf);

% 6: Select the best configuration
% rule: highest malignant recall among config with >= 80% precision
% (ties go to the 1st config in the grid order)
eval_rbf = results_rbf(results_rbf.Precision >= 0.80, :);
if isempty(eval_rbf)
    disp("No setting reached at least 80% precision.")
    eval_rbf = results_rbf; % fallback to all configs
end
[~, best_idx] = max(eval_rbf.Recall);
best_rbf = eval_rbf(best_idx, :);
disp("Best RBF SVM setting:");
disp(best_rbf);

% 7: Save results
writetable(results_rbf, "results/tune_rbf_" + run_num + ".csv");

% recall heatmap
figure;
h = heatmap(string(box), string(kernel), recall_grid);
h.CellLabelFormat = '%.1f';
xlabel('Box Constraint');
ylabel('Kernel Scale');
title('RBF SVM malignant recall, 5-fold CV');
saveas(gcf, "results/figures/rbf_recall_heatmap_" + run_num + ".png");

% recall vs. precision for all configs, coloured by kernel scale
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