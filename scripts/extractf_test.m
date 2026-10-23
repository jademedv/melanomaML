% 1: Load test images
imds_test = imageDatastore('/Users/jade/Library/CloudStorage/OneDrive-UTS/melanoma_cancer_dataset/test', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});

% 2: Remove duplicate images from the test set
dupes = readtable('results/duplicate_check.csv'); % created by check_duplicates.m, images matched by filename
[~, test_base, test_ext] = fileparts(string(imds_test.Files));
keep = ~ismember(test_base + test_ext, string(dupes.TestImage));
imds_test = subset(imds_test, find(keep));
num_test = numel(imds_test.Files);
fprintf("Feature extraction on %d images...\n", num_test);

% 3: Preallocate feature arrays to one row per image
% colour features, mean intensity of each channel
meanR = zeros(num_test, 1);
meanG = zeros(num_test, 1);
meanB = zeros(num_test, 1);
% texture features
contrast = zeros(num_test, 1);
correlation = zeros(num_test, 1);
energy = zeros(num_test, 1);
homogeneity = zeros(num_test, 1);

% 4: Extract features from each images; same procedure as training set
for i = 1:num_test
    img = readimage(imds_test, i);
    % mean pixel value of RGB channels
    meanR(i) = mean(img(:,:,1), "all");
    meanG(i) = mean(img(:,:,2), "all");
    meanB(i) = mean(img(:,:,3), "all");
    % grey level co-occurrence matrix from greyscale img + GLCM properties
    glcm = graycomatrix(rgb2gray(img));
    stats = graycoprops(glcm, {"Contrast", "Correlation", "Energy", "Homogeneity"});
    contrast(i) = stats.Contrast;
    correlation(i) = stats.Correlation;
    energy(i) = stats.Energy;
    homogeneity(i) = stats.Homogeneity;

    if mod(i, 100) == 0 % progress message every 100 images
        fprintf("Processed %d / %d images\n", i, num_test);
    end
end

% 5: Create feature table + save
% row order follows datastore order, which later scripts rely on
test_features = table(meanR, meanG, meanB, contrast, correlation, energy, homogeneity, ...
    'VariableNames', {'MeanR', 'MeanG', 'MeanB', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
test_features.Label = imds_test.Labels; % class label
disp(countcats(test_features.Label)); % num of images per class after duplicate removal
save('data/test_features.mat', 'test_features');
disp("Test feature extraction complete");