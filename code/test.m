%%%%% my test.m file %%%%%

run('vlfeat/toolbox/vl_setup')

[~,~,~] = mkdir('visualizations');

data_path = '../data/'; %change if you want to work with a network copy
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
% test_scn_path = fullfile(data_path,'extra_test_scenes'); %Bonus scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

feature_params = struct('template_size', 36, 'hog_cell_size', 6);

train_images = dir(fullfile(train_path_pos, '*.jpg'));
num_images = size(train_images);
all_images = cell(num_images);
D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
%feature=[];
features_pos = zeros([num_images(1) D]);
% parfor j=1:num_images(1)
for j = 1:5
   all_images{j, 1} = fullfile(train_path_pos, train_images(j).name);
   all_images{j, 1} = single(imread(all_images{j, 1}));
%        if(size(all_images{j,1},3) > 1)
%             all_images{j, 1} = rgb2gray(all_images{j, 1});
%        end
   hog = vl_hog(all_images{j, 1}, feature_params.hog_cell_size);
   size(all_images{j,1})
   size(hog)
   features_pos(j, :) = reshape(hog, 1, D);
end

size(features_pos(j, :))
size(features_pos)

pause;


%initialize these as empty and incrementally expand them.
test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

img = imread( fullfile( test_scn_path, test_scenes(1).name ));
img = single(img)/255;
img_size = size(img)
if(size(img,3) > 1)
    img = rgb2gray(img);
end
length(test_scenes);

%%%%% Last containing variables for whole images
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

%%%%% parameters
hog_cell_size = feature_params.hog_cell_size;
window_size = hog_cell_size^2;
x_step = 6;
y_step = 6;
threshold = 0.1;
scale_step = 0.9;
scale_factor = 1.4;
D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
input_feature = zeros([1 D]);

% loop : images
for i = 1:1
    %%%%% variables for 1 image
    cur_bboxes = zeros(0, 4);
    cur_confidences = zeros(0,1);
    cur_image_ids = cell(0,1);

    %%%%% Loop : scale of image
    while 1
        img_scaled = imresize(img, scale_factor);
        %%%%% parameters for 1 image
        x_size = size(img_scaled, 2)
        y_size = size(img_scaled, 1)
        x_pos = 1;
        y_pos = 1;
        if ( x_size < window_size ) || ( y_size < window_size )
            break;
        end
        x_axis = x_size/x_step;
        y_axis = y_size/y_step;
        %%%%% Loop : y axis
        while ( (y_pos+window_size-1) <= y_size )
            %%%%% Loop : x axis
            while ( (x_pos+window_size-1) <= x_size )

                %%%%% get window sized image from original image
                temp_img = img_scaled(y_pos:y_pos+window_size-1, x_pos:x_pos+window_size-1);
                temp_bboxes = [x_pos, y_pos, x_pos+window_size-1, y_pos+window_size-1];
                size(temp_img);
                %%%%% get hog feature
                hog = vl_hog(temp_img, feature_params.hog_cell_size);
                hog3 = vl_hog(temp_img, 3);
                hog9 = vl_hog(temp_img, 9);

                size(hog); % 6 6 31
                size(hog3); % 12 12 31
                size(hog9); % 4 4 31

                %%%%% make hog feature
                input_feature(1, :) = reshape(hog, 1, D);
                %%%%% compute confidence of this window
                temp_confidence = input_feature*w + b;
                %%%%% combine temp variables to 1 image variables
                if temp_confidence > threshold
                    temp_confidence;
                    cur_bboxes = [cur_bboxes; temp_bboxes];
                    cur_confidences = [cur_confidences; temp_confidence];
                    cur_image_ids = [cur_image_ids; {test_scenes(i).name}];
                end

                x_pos = x_pos + x_step;
            end
            x_pos = 1;
            y_pos = y_pos + y_step;
        end
        scale_factor = scale_factor * scale_step
    end

    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);

    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end

% [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

% cur_confidences = cur_confidences(is_maximum,:);
% cur_bboxes      = cur_bboxes(     is_maximum,:);
% cur_image_ids   = cur_image_ids(  is_maximum,:);

% bboxes      = [bboxes;      cur_bboxes];
% confidences = [confidences; cur_confidences];
% image_ids   = [image_ids;   cur_image_ids];

[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)
    % for j = 1:
    %     %  Loop : y axis
    %     for k = 1:
    % hog = vl_hog(img, feature_params.hog_cell_size);
    % size(hog)
    % input_feature(1, :) = reshape(hog, 1, D);


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

