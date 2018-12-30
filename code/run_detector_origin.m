% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.

%%%%% @params test_scn_path Indicate Test file path
%%%%% @params w trained weights by using svm
%%%%% @params b trained bias by using svm
%%%%% @params feature_params HOG feature parameters ('template_size', 36, 'hog_cell_size', 6)
%%%%% rescale each step of your multiscale detector
%%%%% threshold for a detection
function [bboxes, confidences, image_ids] = ....
    run_detector(test_scn_path, w, b, feature_params)

%%%%% Input Params Info
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

%%%%% Output Info
% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i.
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

img = imread( fullfile( test_scn_path, test_scenes(1).name ));
img = single(img)/255;
if(size(img,3) > 1)
    img = rgb2gray(img);
end
% figure(4);
% imshow(img);
% img_scaled1 = imresize(img, 0.5);
% figure(5);
% imshow(img);
% img_scaled2 = imresize(img, 2);
% figure(6);
% imshow(img);
% img_scaled3 = imresize(img, 3);
% figure(7);
% imshow(img);
% img_scaled4 = imresize(img, 4);
% figure(8);
% imshow(img);
% figure(14);
% imshow(img_scaled1);
% figure(15);
% imshow(img_scaled2);
% figure(16);
% imshow(img_scaled3);
% figure(17);
% imshow(img_scaled4);

size(img)
size(img, 3)
size(img, 2)
size(img, 1)
length(test_scenes)

for i = 1:length(test_scenes)

    % fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end

    %------------------------------------------------------------------------------------------------
    % You have to change the code here.
    % This is the example for creating 15 random detections per image.
    % You have to create a sliding window based multi-scale detection.

    %%%%%%

    cur_x_min = rand(15,1) * size(img,2);
    cur_y_min = rand(15,1) * size(img,1);
    cur_bboxes = [cur_x_min, cur_y_min, cur_x_min + rand(15,1) * 50, cur_y_min + rand(15,1) * 50];
    cur_confidences = rand(15,1) * 2 - 1; %confidences in the range [-2 2]
    cur_image_ids(1:15,1) = {test_scenes(i).name};

    %%%%%%


    %------------------------------------------------------------------------------------------------

    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);

    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];

end



%%%%% we might be able to show hog feature image according to
%%%%% the number of orientations.
% Specify the number of orientations
% hog = vl_hog(im, cellSize, 'verbose', 'numOrientations', o) ;
% imhog = vl_hog('render', hog, 'verbose', 'numOrientations', o) ;

    % cur_x_min = rand(15,1) * size(img,2);
    % cur_y_min = rand(15,1) * size(img,1);
    % cur_bboxes = [cur_x_min, cur_y_min, cur_x_min + rand(15,1) * 50, cur_y_min + rand(15,1) * 50];
    % cur_confidences = rand(15,1) * 2 - 1; %confidences in the range [-2 2]
    % cur_image_ids(1:15,1) = {test_scenes(i).name};