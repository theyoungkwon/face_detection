% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

    image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
    num_images = length(image_files);
    D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
    all_images = cell(num_images);
    features_neg = zeros([num_samples D]);
    smp_per_img=175;
    window_size_cell = (feature_params.template_size / feature_params.hog_cell_size);
    i=1;
    for j=1:num_images(1)
       all_images{j, 1} = fullfile(non_face_scn_path, image_files(j).name);
       all_images{j, 1} = single(imread(all_images{j, 1}));
%        if(size(all_images{j,1},3) > 1)
%             all_images{j, 1} = rgb2gray(all_images{j, 1});
%        end
       num_sampd = 0;
       freq = 4;
       hog = vl_hog(all_images{j, 1}, feature_params.hog_cell_size);
       temp_size = size(hog);
       for y=1:(temp_size(1) - (window_size_cell-1) )/freq
           for x=1:(temp_size(2) - (window_size_cell-1) )/freq
               xStart = x*freq;
               yStart = y*freq;
               xEnd = xStart + (window_size_cell - 1);
               yEnd = yStart + (window_size_cell - 1);
               frame = hog(yStart:yEnd, xStart:xEnd, :);
               features_neg(i, :) = reshape(frame, 1, D);
               i = i+1;
               num_sampd=num_sampd+1;
               % fprintf('  num_sampd:   %d\n', num_sampd);
               if (i==num_samples)
                   j
                   return;
               end
               if (num_sampd >= num_samples/smp_per_img)
                   break;
               end
           end
           if (num_sampd >= num_samples/smp_per_img)
               break;
           end
       end
       if (num_sampd >= num_samples/smp_per_img)
           continue;
       end
    end
    i
    fprintf('  num samples:   %d\n', num_samples);
    fprintf('  i:   %d\n', i);