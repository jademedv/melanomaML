imds_test = imageDatastore('/Users/jade/Library/CloudStorage/OneDrive-UTS/melanoma_cancer_dataset/test', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});


% remove duplicate images from test set
dupes = readtable('results/duplicate_check.csv');
[~, test_base, test_ext] = fileparts(string(imds_test.Files));
keep = ~ismember(test_base + test_ext, string(dupes.TestImage));
imds_test = subset(imds_test, find(keep));
num_test = numel(imds_test.Files);
fprintf("Feature extraction on %d images...\n", num_test);

% RGB
meanR = zeros(num_test, 1);
meanG = zeros(num_test, 1);
meanB = zeros(num_test, 1);

% feature extraction
contrast = zeros(num_test, 1);
correlation = zeros(num_test, 1);
energy = zeros(num_test, 1);
homogeneity = zeros(num_test, 1);

for i = 1:num_test
    img = readimage(imds_test, i);
    meanR(i) = mean(img(:,:,1), "all");
    meanG(i) = mean(img(:,:,2), "all");
    meanB(i) = mean(img(:,:,3), "all");

    glcm = graycomatrix(rgb2gray(img));
    stats = graycoprops(glcm, {"Contrast", "Correlation", "Energy", "Homogeneity"});
    contrast(i) = stats.Contrast;
    correlation(i) = stats.Correlation;
    energy(i) = stats.Energy;
    homogeneity(i) = stats.Homogeneity;

    if mod(i, 100) == 0
        fprintf("Processed %d / %d images\n", i, num_test);
    end
end

test_features = table(meanR, meanG, meanB, contrast, correlation, energy, homogeneity, ...
    'VariableNames', {'MeanR', 'MeanG', 'MeanB', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
test_features.Label = imds_test.Labels;

disp(countcats(test_features.Label));
save('data/test_features.mat', 'test_features');
disp("Test feature extraction complete");