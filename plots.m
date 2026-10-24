% 1: RBF SVM recall heatmap
T = readtable('results/tune_rbf_v1.csv'); % read tuning results
kernel = unique(T.KernelScale);
box_c = unique(T.BoxConstraint);
num_ks = numel(kernel);
num_bc = numel(box_c);
recall_grid = zeros(num_ks, num_bc);
% arrange recall as kernel scale x box constraint
for i = 1:num_ks
    for j = 1:num_bc
        row = T.KernelScale == kernel(i) & T.BoxConstraint == box_c(j);
        recall_grid(i, j) = T.Recall(row) * 100;
    end
end
lo = min(recall_grid, [], 'all');
hi = max(recall_grid, [], 'all');
% selected config = highest recall; ties go to the first one found
best_recall = max(T.Recall);
[~, first_idx] = max(T.Recall);
% heatmap with yellow/green/blue colour map, interpolated to 256 levels
figure('Position', [100 100 800 560]);
imagesc(recall_grid);
ax = gca;
ax.YDir = 'normal';
ax.CLim = [lo hi];
stops = [255 255 217; 199 233 180; 127 205 187; 65 182 196; 29 145 192; 34 94 168; 12 44 132] / 255;
colormap(ax, interp1(linspace(0, 1, size(stops, 1)), stops, linspace(0, 1, 256)));
c = colorbar;
c.Label.String = 'Malignant recall (%)';
xticks(1:num_bc);
xticklabels(string(box_c));
yticks(1:num_ks);
yticklabels(string(kernel));
xlabel('Box constraint');
ylabel('Kernel scale');
ax.FontSize = 11;
ax.TickLength = [0 0];
% cell labels, white text on dark cells, black test on light cells
for i = 1:num_ks
    for j = 1:num_bc
        if recall_grid(i, j) > lo + 0.55 * (hi - lo)
            label_colour = 'w';
        else
            label_colour = 'k';
        end
        text(j, i, sprintf('%.2f', recall_grid(i, j)), 'HorizontalAlignment', 'center', ...
            'Color', label_colour, 'FontSize', 9);
    end
end
% outline the selected configuration (solid) and any tied configurations (dashed)
for r = 1:height(T)
    if abs(T.Recall(r) - best_recall) < 1e-9
        j = find(box_c == T.BoxConstraint(r));
        i = find(kernel == T.KernelScale(r));
        if r == first_idx
            line_style = '-';
        else
            line_style = '--';
        end
        rectangle('Position', [j - 0.5, i - 0.5, 1, 1], 'EdgeColor', [0.84 0.1 0.1], ...
            'LineWidth', 2.5, 'LineStyle', line_style);
    end
end
title('RBF SVM malignant recall (5-fold CV)');
exportgraphics(gcf, 'results/figures/rbf_recall_heatmap_adjusted.png', 'Resolution', 300);

% 2: k-NN recall heatmap
T = readtable('results/tune_knn_v1.csv'); % read tuning results
dist_col = string(T.Distance);
weight_col = string(T.Weight);
k_vals = unique(T.K);
dist_names = ["euclidean", "cityblock", "cosine"];
dist_labels = ["Euclidean", "Cityblock", "Cosine"];
weight_names = ["equal", "inverse", "squaredinverse"];
weight_labels = ["Equal", "Inverse", "Squared inverse"];
num_k = numel(k_vals);
num_w = numel(weight_names);
num_d = numel(dist_names);
% recall: rows = weighting, columns = k, third dimension = distance
recall_grid = zeros(num_w, num_k, num_d);
for d = 1:num_d
    for w = 1:num_w
        for j = 1:num_k
            row = dist_col == dist_names(d) & weight_col == weight_names(w) & T.K == k_vals(j);
            recall_grid(w, j, d) = T.Recall(row) * 100;
        end
    end
end
% one colour range shared by all panels
lo = min(recall_grid, [], 'all');
hi = max(recall_grid, [], 'all');
% selected configuration
[~, best_idx] = max(T.Recall);
best_d = find(dist_names == dist_col(best_idx));
best_w = find(weight_names == weight_col(best_idx));
best_j = find(k_vals == T.K(best_idx));
% colour map (same as RBF heatmap)
stops = [255 255 217; 199 233 180; 127 205 187; 65 182 196; 29 145 192; 34 94 168; 12 44 132] / 255;
cmap = interp1(linspace(0, 1, size(stops, 1)), stops, linspace(0, 1, 256));
% one heatmap panel per distance measure
figure('Position', [100 100 1300 420]);
tl = tiledlayout(1, num_d, 'TileSpacing', 'compact');
for d = 1:num_d
    ax = nexttile;
    imagesc(recall_grid(:, :, d));
    ax.YDir = 'normal';
    ax.CLim = [lo hi];
    colormap(ax, cmap);
    xticks(1:num_k);
    xticklabels(string(k_vals));
    yticks(1:num_w);
    if d == 1 % weighting labels on first panel only
        yticklabels(weight_labels);
        ylabel('Distance weighting');
    else
        yticklabels({});
    end
    xlabel('Number of neighbours (k)');
    title(dist_labels(d));
    ax.FontSize = 11;
    ax.TickLength = [0 0];
    % cell labels
    for w = 1:num_w
        for j = 1:num_k
            if recall_grid(w, j, d) > lo + 0.55 * (hi - lo)
                label_colour = 'w';
            else
                label_colour = 'k';
            end
            text(j, w, sprintf('%.2f', recall_grid(w, j, d)), 'HorizontalAlignment', 'center', ...
                'Color', label_colour, 'FontSize', 9);
        end
    end
    % outline the selected configuration
    if d == best_d
        rectangle('Position', [best_j - 0.5, best_w - 0.5, 1, 1], 'EdgeColor', [0.84 0.1 0.1], ...
            'LineWidth', 2.5);
    end
end
c = colorbar;
c.Layout.Tile = 'east';
c.Label.String = 'Malignant recall (%)';
title(tl, 'k-NN malignant recall (5-fold CV)');
exportgraphics(gcf, 'results/figures/knn_recall_heatmap_adjusted.png', 'Resolution', 300);

% 3: Recall vs. precision scatter
rbf = readtable('results/tune_rbf_v1.csv');
knn = readtable('results/tune_knn_v1.csv');
blue = [0 0.447 0.698]; % RBF SVM
orange = [0.835 0.369 0]; % k-NN
[~, rbf_best] = max(rbf.Recall);
[~, knn_best] = max(knn.Recall);
figure('Position', [100 100 800 560]);
hold on;
% every tuned config
h1 = scatter(rbf.Precision * 100, rbf.Recall * 100, 40, blue, 'o', 'filled', ...
    'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', 'none');
h2 = scatter(knn.Precision * 100, knn.Recall * 100, 40, orange, 's', 'filled', ...
    'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', 'none');
% selected configurations, larger markers with black outline
scatter(rbf.Precision(rbf_best) * 100, rbf.Recall(rbf_best) * 100, 140, blue, 'o', 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.8);
scatter(knn.Precision(knn_best) * 100, knn.Recall(knn_best) * 100, 140, orange, 's', 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.8);
xline(80, '--k', '80% precision floor', 'LabelOrientation', 'horizontal', ...
    'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'right', 'FontSize', 9);
% labels for the selected configurations
% leader lines + text positions were placed manually
plot([81.9 rbf.Precision(rbf_best) * 100], [87.5 rbf.Recall(rbf_best) * 100], 'k-', 'LineWidth', 0.8);
text(80.3, 87.55, {'RBF SVM selected', ...
    sprintf('kernel scale %g, box constraint %g', rbf.KernelScale(rbf_best), rbf.BoxConstraint(rbf_best))}, ...
    'VerticalAlignment', 'bottom', 'FontSize', 9);
plot([84.3 knn.Precision(knn_best) * 100], [87.5 knn.Recall(knn_best) * 100], 'k-', 'LineWidth', 0.8);
text(84.0, 87.55, {'k-NN selected', ...
    sprintf('k = %d, %s, %s', knn.K(knn_best), string(knn.Distance(knn_best)), string(knn.Weight(knn_best)))}, ...
    'VerticalAlignment', 'bottom', 'FontSize', 9);
hold off;
xlim([79 87]);
ylim([80 88.5]);
xlabel('Malignant precision (%)');
ylabel('Malignant recall (%)');
legend([h1 h2], {'RBF SVM', 'k-NN'}, 'Location', 'northeast');
ax = gca;
ax.FontSize = 11;
ax.Box = 'off';
grid(ax, 'on');
ax.GridAlpha = 0.15;
title('Recall and precision for RBF SVM & k-NN (5-fold CV)');
exportgraphics(gcf, 'results/figures/tuning_scatter_adjusted.png', 'Resolution', 300);

% 4: Feature group comparison, malignant recall with 95% CIs
rbf = readtable('results/feature_comparison_rbf.csv');
knn = readtable('results/feature_comparison_knn.csv');
blue = [0 0.447 0.698];
orange = [0.835 0.369 0];
% columns = models, rows = feature sets; converted to %
recall = [rbf.Recall knn.Recall] * 100;
lo = [rbf.RecallLo knn.RecallLo] * 100;
hi = [rbf.RecallHi knn.RecallHi] * 100;
figure('Position', [100 100 700 480]);
b = bar(recall, 'grouped');
b(1).FaceColor = blue;
b(2).FaceColor = orange;
hold on;
for k = 1:2
    % error bars show 95% Clopper-Pearson interval, value printed above
    % each bar
    errorbar(b(k).XEndPoints, recall(:, k), recall(:, k) - lo(:, k), hi(:, k) - recall(:, k), ...
        'k', 'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 8);
    text(b(k).XEndPoints, hi(:, k) + 1.5, compose('%.1f', recall(:, k)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end
yline(80, '--k', '80% target', 'LabelHorizontalAlignment', 'left', 'FontSize', 9);
hold off;
xticks(1:height(rbf));
xticklabels(rbf.FeatureSet);
ylim([0 100]);
xlabel('Feature set');
ylabel('Malignant recall (%)');
legend(b, {'RBF SVM', 'k-NN'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
ax = gca;
ax.FontSize = 11;
ax.Box = 'off';
title('Malignant recall by feature set (5-fold CV)');
exportgraphics(gcf, 'results/figures/feature_groups_recall.png', 'Resolution', 300);

% 5: Robustness figures: recall and specificity vs severity
T = readtable('results/robustness_results.csv');
T.Model = string(T.Model);
T.Perturbation = string(T.Perturbation);
models = unique(T.Model, 'stable');
perts = ["Gaussian noise","Gaussian blur","Brightness","Contrast","Red channel"];
units = ["variance","sigma","offset","factor","scale"];
colors = [0 0.447 0.698; 0.835 0.369 0];   % SVM blue, k-NN vermilion
markers = {'o','s'};
% clean test set baselines (copy from results/test_results.csv)
% rows = models in the order of unique(T.Model), columns = recall, specificity
baseline = [0.8869 0.829;
    0.8848 0.847];
% one figure per metric
metrics = ["Recall","Specificity"];
labels = ["Malignant recall","Specificity"];
files = ["robustness_recall.png","robustness_specificity.png"];
for m = 1:2
    fig = figure('Color','w','Position',[100 100 1100 650]);
    tl = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
    title(tl, labels(m) + " under input perturbations", 'FontWeight','bold');
    % one panel per perturbation
    for p = 1:5
        ax = nexttile; hold(ax,'on');
        for k = 1:numel(models)
            rows = T.Model == models(k) & T.Perturbation == perts(p);
            sub = sortrows(T(rows,:), 'Level');
            % dashed line = clean baseline, solid line = perturbed results
            yline(ax, baseline(k,m), '--', 'Color', colors(k,:), 'LineWidth', 0.8, 'HandleVisibility','off');
            plot(ax, sub.Level, sub.(metrics(m)), ['-' markers{k}], ...
                'Color', colors(k,:), 'MarkerFaceColor', colors(k,:), ...
                'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', models(k));
        end
        title(ax, perts(p));
        xlabel(ax, units(p));
        ylabel(ax, labels(m));
        ylim(ax, [0 1]);
        grid(ax,'on'); box(ax,'off');
    end
    % legend placed in the empty tile
    lgd = legend(ax, 'Location','northwest');
    lgd.Layout.Tile = 6;
    exportgraphics(fig, "results/figures/" + files(m), 'Resolution', 300);
end

% 6: Class distributions of the colour features (training set)
load('data/train_features.mat');
label = train_features.Label;
channels = {'MeanR','MeanG','MeanB'};
titles = {'Red','Green','Blue'};
benign_color = [0 0.447 0.698];       % blue
malignant_color = [0.835 0.369 0];    % vermilion
edges = 0:5:255; % histogram bins of 5 intensity levels
fig = figure('Color','w','Position',[100 100 1100 380]);
tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
title(tl, 'Mean channel intensity by class (training set)', 'FontWeight','bold');
% one panel per colour channel
for c = 1:3
    ax = nexttile; hold(ax,'on');
    x_benign = train_features.(channels{c})(label == 'benign');
    x_malignant = train_features.(channels{c})(label == 'malignant');
    % densities (area = 1) so classes are comparable regardless of size
    histogram(ax, x_benign, edges, 'Normalization','pdf', ...
        'FaceColor', benign_color, 'FaceAlpha', 0.5, 'EdgeColor','none', ...
        'DisplayName','Benign');
    histogram(ax, x_malignant, edges, 'Normalization','pdf', ...
        'FaceColor', malignant_color, 'FaceAlpha', 0.5, 'EdgeColor','none', ...
        'DisplayName','Malignant');
    % dashed vertical lines mark the class means
    xline(ax, mean(x_benign), '--', 'Color', benign_color, 'LineWidth', 1.2, 'HandleVisibility','off');
    xline(ax, mean(x_malignant), '--', 'Color', malignant_color, 'LineWidth', 1.2, 'HandleVisibility','off');
    title(ax, titles{c});
    xlabel(ax, 'Mean intensity (0-255)');
    if c == 1
        ylabel(ax, 'Density');
    end
    xlim(ax, [0 255]);
    grid(ax,'on'); box(ax,'off');
end
legend(ax, 'Location','northwest');
exportgraphics(fig, 'results/figures/feature_distributions.png', 'Resolution', 300);