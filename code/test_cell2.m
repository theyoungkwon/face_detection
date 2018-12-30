n_hog_cells = sqrt(length(w) / 31); %specific to default HoG parameters
fprintf('  n_hog_cells :   %d\n', n_hog_cells);
fprintf('  length(w) :   %d\n', length(w) );
imhog = vl_hog('render', single(reshape(w, [n_hog_cells n_hog_cells 31])), 'verbose') ;
figure(3); imagesc(imhog) ; colormap gray; set(3, 'Color', [.988, .988, .988])   %% Disply HOG  %%

pause(0.1) %let's ui rendering catch up
hog_template_image = frame2im(getframe(3));
% getframe() is unreliable. Depending on the rendering settings, it will
% grab foreground windows instead of the figure in question. It could also
% return a partial image.
imwrite(hog_template_image, 'visualizations/hog_template_lambda00001_cell2.png')
pause(0.1);

%% Step 4. Run detector on test set.
% YOU CODE 'run_detector'. Make sure the outputs are properly structured!
% They will be interpreted in Step 6 to evaluate and visualize your
% results. See run_detector.m for more details.

%%%%% @params test_scn_path Indicate Test file path
%%%%% @params w trained weights by using svm
%%%%% @params b trained bias by using svm
%%%%% @params feature_params HOG feature parameters ('template_size', 36, 'hog_cell_size', 6)
%%%%% rescale each step of your multiscale detector
%%%%% threshold for a detection
[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params);
% [bboxes, confidences, image_ids] = run_detector_origin(test_scn_path, w, b, feature_params);
% [bboxes, confidences, image_ids] = run_detector_from_src(test_scn_path, w, b, feature_params);

% run_detector will have (at least) two parameters which can heavily
% influence performance -- how much to rescale each step of your multiscale
% detector, and the threshold for a detection. If your recall rate is low
% and your detector still has high precision at its highest recall point,
% you can improve your average precision by reducing the threshold for a
% positive detection.


%% Step 5. Evaluate and Visualize detections
% These functions require ground truth annotations, and thus can only be
% run on the CMU+MIT face test set. Use visualize_detectoins_by_image_no_gt
% for testing on extra images (it is commented out below).

% Don't modify anything in 'evaluate_detections'!
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections(bboxes, confidences, image_ids, label_path);

visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp, test_scn_path, label_path)
% visualize_detections_by_image_no_gt(bboxes, confidences, image_ids, test_scn_path)

% visualize_detections_by_confidence(bboxes, confidences, image_ids, test_scn_path, label_path);



% performance to aim for
% random (stater code) 0.001 AP
% single scale ~ 0.2 to 0.4 AP
% multiscale, 6 pixel step ~ 0.83 AP
% multiscale, 4 pixel step ~ 0.89 AP
% multiscale, 3 pixel step ~ 0.92 AP