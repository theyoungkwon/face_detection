% Starter code prepared by James Hays for CS 143, Brown University
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples.

function features_pos = get_positive_features(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
    train_images = dir(fullfile(train_path_pos, '*.jpg'));
    num_images = size(train_images);
    all_images = cell(num_images);
    D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
    %feature=[];
    features_pos = zeros([num_images(1) D]);
    parfor j=1:num_images(1)
       all_images{j, 1} = fullfile(train_path_pos, train_images(j).name);
       all_images{j, 1} = single(imread(all_images{j, 1}));
%        if(size(all_images{j,1},3) > 1)
%             all_images{j, 1} = rgb2gray(all_images{j, 1});
%        end
       hog = vl_hog(all_images{j, 1}, feature_params.hog_cell_size);
       % size(features_pos(j, :))
       % size(features_pos)
       features_pos(j, :) = reshape(hog, 1, D);
       % hog = hog(:)';
       %feature = [feature;hog];
       % size(features_pos)
       
    end
% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray
