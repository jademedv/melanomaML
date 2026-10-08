imds_train = imageDatastore('/Users/jade/Library/CloudStorage/OneDrive-UTS/melanoma_cancer_dataset/train', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});

num_train = numel(imds_train.Files);
fprintf("Feature extraction on %d images...\n", num_train);

% RGB
meanR = zeros(num_train, 1);
meanG = zeros(num_train, 1);
meanB = zeros(num_train, 1);

% feature extraction
contrast = zeros(num_train, 1);
correlation = zeros(num_train, 1);
energy = zeros(num_train, 1);
homogeneity = zeros(num_train, 1);

for i = 1:num_train
    img = readimage(imds_train, i);
    meanR(i) = mean(img(:,:,1), "all");
    meanG(i) = mean(img(:,:,2), "all");
    meanB(i) = mean(img(:,:,3), "all");

    glcm = graycomatrix(rgb2gray(img));
    stats = graycoprops(glcm, {"Contrast", "Correlation", "Energy", "Homogeneity"});
    contrast(i) = stats.Contrast;
    correlation(i) = stats.Correlation;
    energy(i) = stats.Energy;
    homogeneity(i) = stats.Homogeneity;

    if mod(i, 200) == 0
        fprintf("Processed %d / %d images\n", i, num_train);
    end
end

train_features = table(meanR, meanG, meanB, contrast, correlation, energy, homogeneity, ...
    'VariableNames', {'MeanR', 'MeanG', 'MeanB', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
train_features.Label = imds_train.Labels;

disp("Feature extraction complete")
save('data/train_features.mat', 'train_features');

% save classification learner model results
save('rbfSVM.mat','rbfSVM');
save('kNN.mat','kNN');
save('linearSVM.mat','linearSVM');
save('randomForest.mat','randomForest');
