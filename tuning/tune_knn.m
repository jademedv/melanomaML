load('data/train_features.mat');
predictors = train_features(:, 1:end-1);
response = train_features.Label;
class = categorical({'benign'; 'malignant'});
seed = 1;
run_id = 1;
rng(seed);
cvp = cvpartition(response, 'KFold', 5);
run_num = "v" + run_id;

% 1: Define hyperparameter grid
k_values = [50 60 65 70 75 80 85 100]; % number of neighbours
dist = {'euclidean', 'cityblock', 'cosine'}; % distance measures
weights = {'equal', 'inverse', 'squaredinverse'}; % neighbour weighting schemes
num_k = numel(k_values);
num_dist = numel(dist);
num_w = numel(weights);
num_runs = num_k * num_dist * num_w; % total num of configs

% 2: Preallocate result arrays
k_col = zeros(num_runs, 1);
dist_col = strings(num_runs, 1);
weight_col = strings(num_runs, 1);
accuracy = zeros(num_runs, 1);
recall = zeros(num_runs, 1);
precision = zeros(num_runs, 1);
specificity = zeros(num_runs, 1);
f1 = zeros(num_runs, 1);
auc = zeros(num_runs, 1);
recall_lo = zeros(num_runs, 1); % lower bound of recall CI
recall_hi = zeros(num_runs, 1); % upper bound of recall CI
recall_grid = zeros(num_k, num_dist, num_w); % recall by k, distance + weighting for line charts

% 3: Grid search with 5-fold CV
idx = 0;
for d = 1:num_dist
for w = 1:num_w
for i = 1:num_k
        idx = idx + 1;
        fprintf("Run %d / %d: k %d, %s, %s\n", ...
            idx, num_runs, k_values(i), dist{d}, weights{w});
        % train k-NN classifier on each fold
        model = fitcknn(predictors, response, ...
            'NumNeighbors', k_values(i), ...
            'Distance', dist{d}, ...
            'DistanceWeight', weights{w}, ...
            'Standardize', true, ...
            'ClassNames', class, ...
            'CVPartition', cvp);
        % out of fold predictions + scores for every image
        [prediction, score] = kfoldPredict(model);
        % confusion matrix (rows = true class, columns = predicted)
        C = confusionmat(response, prediction, 'Order', class);
        tn = C(1,1);
        fp = C(1,2);
        fn = C(2,1);
        tp = C(2,2);
        % store the config + metrics
        k_col(idx) = k_values(i);
        dist_col(idx) = dist{d};
        weight_col(idx) = weights{w};
        accuracy(idx) = (tp + tn) / sum(C, 'all');
        [recall(idx), ci] = binofit(tp, tp + fn);
        recall_lo(idx) = ci(1);
        recall_hi(idx) = ci(2);
        precision (idx) = tp / (tp + fp);
        specificity(idx) = tn / (tn + fp);
        f1(idx) = 2 * precision(idx) * recall(idx) / (precision(idx) + recall(idx));
        [~, ~, ~, auc(idx)] = perfcurve(response, score(:, 2), 'malignant');
        recall_grid(i, d, w) = recall(idx) * 100;
end
end
end

% 4: Collect results into a table
results_knn = table(k_col, dist_col, weight_col, accuracy, recall, recall_lo, recall_hi, precision, specificity, f1, auc, ...
    'VariableNames', {'K', 'Distance', 'Weight', 'Accuracy', ...
    'Recall', 'RecallLo', 'RecallHi', 'Precision', 'Specificity', 'F1', 'AUC'});
disp(results_knn);

% 5: Select best config
% rule: highest malignant recall among configs with >= 80% precision
% (ties go to the first config in the grid order)
eval_knn = results_knn(results_knn.Precision >= 0.80, :);
if isempty(eval_knn)
    disp("No setting reached at least 80% precision.")
    eval_knn = results_knn;
end
[~, best_idx] = max(eval_knn.Recall);
best_knn = eval_knn(best_idx, :);
disp("Best k-NN setting:");
disp(best_knn);

% 6: Save results
writetable(results_knn, "results/tune_knn_" + run_num + ".csv");
% recall against k: one panel per distance measure, ...
% one line per weighting scheme
figure;
tiledlayout(1, num_dist, 'TileSpacing', 'compact');
for d = 1:num_dist
    nexttile;
    hold on;
for w = 1:num_w
    plot(k_values, recall_grid(:, d, w), '-o');
end
    hold off;
    set(gca, 'XScale', 'log');
    title(dist{d});
    xlabel('Number of neighbours (k)');
    ylabel('Malignant recall');
    legend(weights, 'Location', 'best');
end
sgtitle('k-NN malignant recall, 5-fold CV');
saveas(gcf, "results/figures/knn_recall_lines_" + run_num + ".png");
%recall vs. precision for all configs, coloured by k
figure;
scatter(results_knn.Precision * 100, results_knn.Recall * 100, 40, log10(results_knn.K), 'filled');
hold on;
xline(80, '--k'); % 80% precision benchmark
x = results_knn.Precision * 100;
y = results_knn.Recall * 100;
valid = ~isnan(x) & ~isnan(y);
p = polyfit(x(valid), y(valid), 1); % straight line fit, shown as descriptive trend only
xfit = linspace(min(x(valid)), max(x(valid)), 100);
plot(xfit, polyval(p, xfit), 'r-', 'LineWidth', 0.8);
hold off;
colorbar;
xlabel('Precision (%)');
ylabel('Recall (%)');
title('k-NN recall vs precision');
saveas(gcf, "results/figures/knn_recallprec_" + run_num + ".png");