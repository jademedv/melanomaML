% 1: Load data + final models
addpath('scripts/');
load('data/test_features.mat');
load('models/trained/final_rbfSVM.mat');
load('models/trained/final_knn.mat');
test_labels = test_features.Label;
feature_names = {'MeanR', 'MeanG', 'MeanB', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'};
models = {rbfSVM, kNN};
model_names = ["RBF SVM", "k-NN"];
num_models = numel(models);

% 2: Load same test images as clean evaluation (duplicates excluded)
test_path = '/Users/jade/Library/CloudStorage/OneDrive-UTS/melanoma_cancer_dataset/test';
imds_test = imageDatastore(test_path, ...
'IncludeSubfolders', true, ...
'LabelSource', 'foldernames', ...
'FileExtensions', {'.jpg'});
dupes = readtable('results/duplicate_check.csv');
[~, test_base, test_ext] = fileparts(string(imds_test.Files));
keep = ~ismember(test_base + test_ext, string(dupes.TestImage));
imds_test = subset(imds_test, find(keep));
imgs = readall(imds_test);
num_test = numel(imgs);
if num_test ~= height(test_features)
    error("Image count does not match test_features");
end

% 3: Perturbations + severity levels (fixed before running)
% one row of levels per perturbation, in the same order as pert_names
pert_names = ["Gaussian noise", "Gaussian blur", "Brightness", "Contrast", "Red channel"];
pert_units = ["variance", "sigma", "offset", "factor", "scale"];
levels = [0.001 0.005 0.01 0.02; % noise variance (intensities scaled to 0-1)
          0.5   1     2    3; % blur sigma in pixels
          -40   -20   20   40; % brightness offset in grey levels (0-255)
          0.6   0.8   1.2  1.4; % contrast factor
          0.8   0.9   1.1  1.2]; % red channel scale factor
num_perts = numel(pert_names);
num_levels = size(levels, 2);
% example images for the figure showing what each perturbation does (first = benign, last = malignant)
example_idx = [1 num_test];
example_names = ["benign", "malignant"];
examples = cell(num_perts, num_levels, 2);

% 4: Table from saved test features with no perturbation
results = table();
for m = 1:num_models
    [prediction, score] = predict(models{m}, test_features(:, 1:end-1));
    metrics = calculate_metrics(test_labels, prediction, score(:, 2));
    metrics.Model = model_names(m);
    metrics.Perturbation = "None";
    metrics.Level = NaN;
    results = [results; metrics];
end

% 5: Perturbed test sets
for p = 1:num_perts
    for l = 1:num_levels
        fprintf("%s, %s = %g\n", pert_names(p), pert_units(p), levels(p, l));
        feats = zeros(num_test, 7);
        for i = 1:num_test
            img = imgs{i};
            % apply perturbations, results clipped to 0-255 by the uint8
            % conversion
            if p == 1
                img = imnoise(img, 'gaussian', 0, levels(p, l)); % gaussian noise
            elseif p == 2
                img = imgaussfilt(img, levels(p, l)); % gaussian blur
            elseif p == 3
                img = uint8(double(img) + levels(p, l)); % brightness offset
            elseif p == 4
                img = uint8((double(img) - 128) * levels(p, l) + 128); % contrast scaling
            elseif p == 5
                img(:,:,1) = uint8(double(img(:,:,1)) * levels(p, l)); % red channel scaling
            end
            % keep perturbed example images for the figure
            if i == example_idx(1)
                examples{p, l, 1} = img;
            elseif i == example_idx(2)
                examples{p, l, 2} = img;
            end
            % recompute the same seven images as the original extraction
            feats(i, 1) = mean(img(:,:,1), "all");
            feats(i, 2) = mean(img(:,:,2), "all");
            feats(i, 3) = mean(img(:,:,3), "all");
            glcm = graycomatrix(rgb2gray(img));
            stats = graycoprops(glcm, {"Contrast", "Correlation", "Energy", "Homogeneity"});
            feats(i, 4) = stats.Contrast;
            feats(i, 5) = stats.Correlation;
            feats(i, 6) = stats.Energy;
            feats(i, 7) = stats.Homogeneity;
        end
        % predict with both final models + record the metrics
        perturbed = array2table(feats, 'VariableNames', feature_names);
        for m = 1:num_models
            [prediction, score] = predict(models{m}, perturbed);
            metrics = calculate_metrics(test_labels, prediction, score(:, 2));
            metrics.Model = model_names(m);
            metrics.Perturbation = pert_names(p);
            metrics.Level = levels(p, l);
            results = [results; metrics];
        end
    end
end

% 6: Save results
results = movevars(results, {'Model', 'Perturbation', 'Level'}, 'Before', 1);
disp(results);
writetable(results, 'results/robustness_results.csv');
% 7: Figure for what each perturbation does to one benign + one malignant
% example
for e = 1:2
    figure('Position', [100 100 1100 900]);
    tiledlayout(num_perts, num_levels + 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for p = 1:num_perts
        nexttile;
        imshow(imgs{example_idx(e)});
        title("Original", 'FontSize', 8);
        for l = 1:num_levels
            nexttile;
            imshow(examples{p, l, e});
            title(sprintf('%s\n%s = %g', pert_names(p), pert_units(p), levels(p, l)), 'FontSize', 8);
        end
    end
    exportgraphics(gcf, "results/figures/robustness_examples_" + example_names(e) + ".png", 'Resolution', 300);
end
disp("Robustness test complete");