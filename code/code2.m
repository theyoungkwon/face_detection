
%%%%% the number of hog cells in 1 hog template
window_size_cell = (feature_params.template_size / feature_params.hog_cell_size);
for y=1:(temp_size(1) - (window_size_cell-1) )/freq
  for x=1:(temp_size(2) - (window_size_cell-1) )/freq
    xStart = x*freq;
    yStart = y*freq;
    xEnd = xStart + (window_size_cell - 1);
    yEnd = yStart + (window_size_cell - 1);

