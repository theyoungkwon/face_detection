
%%%%% Set Hog Parameter
feature_params = struct('template_size', 36, 'hog_cell_size', 6);

%%%%% Train Classifier.
[w, b] = vl_svmtrain(X', Y', 0.00001);

%%%%% Run detector on test set.
%%%%% @params test_scn_path Indicate Test file path
%%%%% @params w trained weights by using svm
%%%%% @params b trained bias by using svm
%%%%% @params feature_params HOG feature parameters ('template_size', 36, 'hog_cell_size', 6)
%%%%% rescale each step of your multiscale detector
%%%%% threshold for a detection
[bboxes, confidences, image_ids] = run_detector(test_scn_path, w, b, feature_params);



