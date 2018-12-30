close all
clear
run('vlfeat/toolbox/vl_setup')

[~,~,~] = mkdir('visualizations');

data_path = '../data/'; %change if you want to work with a network copy
train_path_pos = fullfile(data_path, 'caltech_faces/Caltech_CropFaces'); %Positive training examples. 36x36 head crops
non_face_scn_path = fullfile(data_path, 'train_non_face_scenes'); %We can mine random or hard negatives from here
test_scn_path = fullfile(data_path,'test_scenes/test_jpg'); %CMU+MIT test scenes
% test_scn_path = fullfile(data_path,'extra_test_scenes'); %Bonus scenes
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt'); %the ground truth face locations in the test set

feature_params = struct('template_size', 36, 'hog_cell_size', 6);

%% Step 1. Load positive training crops and random negative examples
features_pos = get_positive_features( train_path_pos, feature_params );
num_negative_examples = 60000; %Higher will work strictly better, but you should start with 10000 for debugging
features_neg = get_random_negative_features( non_face_scn_path, feature_params, num_negative_examples);

%% step 2. Train Classifier
Y = -1 * ones([num_negative_examples 1]);
Y = vertcat(Y, ones([size(features_pos, 1), 1]));
X=vertcat(features_neg, features_pos);
[w, b] = vl_svmtrain(X', Y', 0.00001);  %% Training finished here ! -------------------

%% step 3. Examine learned classifier
fprintf('Initial classifier performance on train data:\n')
confidences = [features_pos; features_neg]*w + b;   %% TRAINING DATA AGAIN!
label_vector = [ones(size(features_pos,1),1); -1*ones(size(features_neg,1),1)];
[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy( confidences, label_vector );

% Visualize how well separated the positive and negative examples are at
% training time. Sometimes this can idenfity odd biases in your training
% data, especially if you're trying hard negative mining. This
% visualization won't be very meaningful with the placeholder starter code.
non_face_confs = confidences( label_vector < 0);
face_confs     = confidences( label_vector > 0);
figure(2);
plot(sort(face_confs), 'g'); hold on
plot(sort(non_face_confs),'r');
plot([0 size(non_face_confs,1)], [0 0], 'b');
hold off;
% Visualize the learned detector. This would be a good thing to include in
% your writeup!
n_hog_cells = sqrt(length(w) / 31); %specific to default HoG parameters
fprintf('  n_hog_cells :   %d\n', n_hog_cells);
fprintf('  length(w) :   %d\n', length(w) );
imhog = vl_hog('render', single(reshape(w, [n_hog_cells n_hog_cells 31])), 'verbose') ;
figure(3); imagesc(imhog) ; colormap gray; set(3, 'Color', [.988, .988, .988])   %% Disply HOG  %%

pause(0.1) %let's ui rendering catch up
hog_template_image = frame2im(getframe(3));
imwrite(hog_template_image, 'visualizations/hog_template.png')
pause(0.1);

%% Step 4. Run detector on test set.
%%%%% @params test_scn_path Indicate Test file path
%%%%% @params w trained weights by using svm
%%%%% @params b trained bias by using svm
%%%%% @params feature_params HOG feature parameters ('template_size', 36, 'hog_cell_size', 6)
%%%%% rescale each step of your multiscale detector
%%%%% threshold for a detection
[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params);
% [bboxes, confidences, image_ids] = run_detector_origin(test_scn_path, w, b, feature_params);
% [bboxes, confidences, image_ids] = run_detector_from_src(test_scn_path, w, b, feature_params);

%% Step 5. Evaluate and Visualize detections
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)
