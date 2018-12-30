
%%%%% @params test_scn_path Indicate Test file path
%%%%% @params w trained weights by using svm
%%%%% @params b trained bias by using svm
%%%%% @params feature_params HOG feature parameters ('template_size', 36, 'hog_cell_size', 6)
%%%%% rescale each step of your multiscale detector
%%%%% threshold for a detection
function [bboxes, confidences, image_ids] = ....
    run_detector(test_scn_path, w, b, feature_params)

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%%%%% Last containing variables for whole images
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

%%%%% parameters
template_size = feature_params.template_size;
hog_cell_size = feature_params.hog_cell_size;
window_size_cell = template_size / hog_cell_size;
x_step = 1;
y_step = 1;
threshold = 0.5;
D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
input_feature = zeros([1 D]);

for i = 1:length(test_scenes)

    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end

    %%%%% variables for 1 image
    scale_factor = 1.4;
    cur_bboxes = zeros(0, 4);
    cur_confidences = zeros(0,1);
    cur_image_ids = cell(0,1);

    %%%%% Loop : scale of image
    while 1
        temp_img = imresize(img, scale_factor);
        %%%%% if size of x, y axis of image is larger than template size,
        %%%%% then break
        if ( size(temp_img, 2) < template_size ) || ( size(temp_img, 1) < template_size )
            break;
        end
        %%%%% get hog feature
        hog = vl_hog(temp_img, feature_params.hog_cell_size);
        %%%%% parameters for 1 image
        x_size_cell = size(hog, 2);
        y_size_cell = size(hog, 1);
        x_pos_cell = 1;
        y_pos_cell = 1;
        %%%%% Loop : y axis
        while ( (y_pos_cell+window_size_cell-1) <= y_size_cell )
            %%%%% Loop : x axis
            while ( (x_pos_cell+window_size_cell-1) <= x_size_cell )

                %%%%% get window cell sized hog feature from original hog feature
                temp_hog = hog(y_pos_cell:y_pos_cell+window_size_cell-1, x_pos_cell:x_pos_cell+window_size_cell-1, :);
                %%%%% compute confidence of this window
                temp_confidence = sum(temp_hog(:).*w) + b;

                %%%%% if confidence is bigger than the threshold,
                %%%%% then get this window as a face
                if temp_confidence > threshold
                    temp_confidence;
                    %%%%% set bounding box
                    temp_bboxes = [x_pos_cell, y_pos_cell, x_pos_cell+window_size_cell-1, y_pos_cell+window_size_cell-1];
                    temp_bboxes = temp_bboxes * hog_cell_size / scale_factor;
                    %%%%% append bounding box to cur_bboxes
                    cur_bboxes = [cur_bboxes; temp_bboxes];
                    %%%%% append confidence to cur_confidences
                    cur_confidences = [cur_confidences; temp_confidence];
                    %%%%% append image name to cur_iamge_ids
                    cur_image_ids = [cur_image_ids; {test_scenes(i).name}];
                end

                x_pos_cell = x_pos_cell + x_step;
            end
            x_pos_cell = 1;
            y_pos_cell = y_pos_cell + y_step;
        end
        % scale_factor = scale_factor * scale_step
        scale_factor = scale_factor - 0.1;
    end

    %%%%%%
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);

    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];

End
