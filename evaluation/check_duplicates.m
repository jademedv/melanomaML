train = 'Users/jade/Library/CloudStorage/OneDrive-UTS/melanoma_cancer_dataset/train';
test = 'Users/jade/Library/CloudStorage/OneDrive-UTS/melanoma_cancer_dataset/test';

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
size = 16;
threshold = 0.5;
fprintf("%d train and %d test images\n", num_train, num_test);

% 16x16 grayscale thumbnail per image
train_thumbs = zeros(num_train, size^2);
for i = 1:num_train
    img = readimage(imds_train, i);
    small = imresize(rgb2gray(img), [size size]);
    train_thumbs(i, :) = double(small(:))' / 255;
    if mod(i, 1000) == 0
        fprintf("Train thumbnails: %d / %d\n", i, num_train);
    end
end

test_thumbs = zeros(num_test, size^2);
for i = 1:num_test
    img = readimage(imds_test, i);
    small = imresize(rgb2gray(img), [size size]);
    test_thumbs(i, :) = double(small(:))' / 255;
end