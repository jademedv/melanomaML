% 1: Load train + test images
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

% 2: Settings
imgsize = 16; % side length of thumbnail in pixels
threshold = 0.1; % test images closer than this to a training image are flagged
fprintf("%d train and %d test images\n", num_train, num_test);

% 3: Convert every image to 16x16 greyscale thumbnail
% each thumbnail is flattened to one row
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

% 4: Find nearest training image for every test img
% d_tt = Euclidean distance to nearest training thumbnail
% nn_tt = index of that training image
[d_tt, nn_tt] = pdist2(train_thumbs, test_thumbs, 'euclidean', 'Smallest', 1);
dupe_test = d_tt < threshold; % test images flagged as duplicates

% 5: Save flagged test pairs
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

% 6: Figure of exact duplicates (distance of 0)
% test image top row, training match below
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

% 7: Distribution of nearest neighbour distance for all test images
figure;
histogram(d_tt, 50);
hold on;
xline(threshold, '--r'); % flagging threshold
hold off;
xlabel('Distance to nearest training image');
ylabel('Number of test images');
title('Nearest-neighbour distance (test vs train)');
saveas(gcf, 'results/figures/duplicate_distances_all.png');
writetable(table((1:numel(d_tt))', d_tt(:), 'VariableNames', {'TestIndex', 'Distance'}), ...
    'results/duplicate_distances_all.csv');