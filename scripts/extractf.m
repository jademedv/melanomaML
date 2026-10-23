% 1: Load the training images
imds_train = imageDatastore('/Users/jade/Library/CloudStorage/OneDrive-UTS/melanoma_cancer_dataset/train', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ... % labels are taken from folder names
    'FileExtensions', {'.jpg'});
num_train = numel(imds_train.Files);
fprintf("Feature extraction on %d images...\n", num_train);

% 2: Preallocate feature arrays to one row per image
% define colour features, mean intensity of each channel
meanR = zeros(num_train, 1);
meanG = zeros(num_train, 1);
meanB = zeros(num_train, 1);
% texture features
contrast = zeros(num_train, 1);
correlation = zeros(num_train, 1);
energy = zeros(num_train, 1);
homogeneity = zeros(num_train, 1);

% 3: Extract features from each image
for i = 1:num_train
    img = readimage(imds_train, i);
    % mean pixel value of RGB channels
    meanR(i) = mean(img(:,:,1), "all");
    meanG(i) = mean(img(:,:,2), "all");
    meanB(i) = mean(img(:,:,3), "all");
    % build grey level co-occurrence matrix from greyscale image
    glcm = graycomatrix(rgb2gray(img));
    % compute properties
    stats = graycoprops(glcm, {"Contrast", "Correlation", "Energy", "Homogeneity"});
    contrast(i) = stats.Contrast;
    correlation(i) = stats.Correlation;
    energy(i) = stats.Energy;
    homogeneity(i) = stats.Homogeneity;

    if mod(i, 200) == 0 % progress message in terminal every 200 images
        fprintf("Processed %d / %d images\n", i, num_train);
    end
end

% 4: Create feature table + save
train_features = table(meanR, meanG, meanB, contrast, correlation, energy, homogeneity, ...
    'VariableNames', {'MeanR', 'MeanG', 'MeanB', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
train_features.Label = imds_train.Labels; % class label (benign/malignant)
disp("Feature extraction complete")
save('data/train_features.mat', 'train_features');

% 5: Save Classification Learner model results
% note: these variables were exported from the Classification Learner app ... 
% to the workspace; the variables are not created by script
save('rbfSVM.mat','rbfSVM');
save('kNN.mat','kNN');
save('linearSVM.mat','linearSVM');
save('randomForest.mat','randomForest');
