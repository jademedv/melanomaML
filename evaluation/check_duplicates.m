train = '/Users/jade/Library/CloudStorage/OneDrive-UTS/melanoma_cancer_dataset/train';
test = '/Users/jade/Library/CloudStorage/OneDrive-UTS/melanoma_cancer_dataset/test';

imds_train = imageDatastore(train, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});

imds_test = imageDatastore(test, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});

num_train = numel(imds_train.Files);
num_test = numel(imds_test.Files);
imgsize = 16;
threshold = 0.1;
fprintf("%d train and %d test images\n", num_train, num_test);

% 16x16 grayscale thumbnail per image
train_thumbs = zeros(num_train, imgsize^2);
for i = 1:num_train
    img = readimage(imds_train, i);
    small = imresize(rgb2gray(img), [imgsize imgsize]);
    train_thumbs(i, :) = double(small(:))' / 255;
end

test_thumbs = zeros(num_test, imgsize^2);
for i = 1:num_test
    img = readimage(imds_test, i);
    small = imresize(rgb2gray(img), [imgsize imgsize]);
    test_thumbs(i, :) = double(small(:))' / 255;
end

% nearest train img for every test img
[d_tt, nn_tt] = pdist2(train_thumbs, test_thumbs, 'euclidean', 'Smallest', 1);
dupe_test = d_tt < threshold;

% save flagged test pairs
[~, test_base, test_ext] = fileparts(string(imds_test.Files));
[~, train_base, train_ext] = fileparts(string(imds_train.Files));

test_names = test_base + test_ext;
train_names = train_base + train_ext;

test_idx = find(dupe_test)';
train_idx = nn_tt(dupe_test)';

results_dupe = table(test_idx, train_idx, ...
    test_names(test_idx), train_names(train_idx), ...
    imds_test.Labels(test_idx), imds_train.Labels(train_idx), d_tt(dupe_test)', ...
'VariableNames', {'TestIndex', 'TrainIndex', 'TestImage', 'TrainImage', ...
'TestLabel', 'TrainLabel', 'Distance'});
results_dupe = sortrows(results_dupe, 'Distance');
writetable(results_dupe, 'results/duplicate_check.csv');

% figure
top = results_dupe(results_dupe.Distance < 1e-6, :);
num_top = height(top);
if num_top > 0
    figure('Position', [100 100 min(250 * num_top, 1800) 520]);
    for p = 1:num_top
        subplot(2, num_top, p);
        imshow(readimage(imds_test, top.TestIndex(p)));
        title("test: " + top.TestImage(p), 'Interpreter', 'none', 'FontSize', 8);
        subplot(2, num_top, num_top + p);
        imshow(readimage(imds_train, top.TrainIndex(p)));
        title("train: " + top.TrainImage(p), 'Interpreter', 'none', 'FontSize', 8);
    end
    saveas(gcf, 'results/figures/duplicates.png');
end