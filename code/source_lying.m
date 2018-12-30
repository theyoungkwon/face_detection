% You have to change the code here.
    % This is the example for creating 15 random detections per image.
    % You have to create a sliding window based multi-scale detection.
    cur_x_min = [];
    cur_y_min = [];
    cur_bboxes = zeros(0,4);
    cur_confidences = zeros(0,1);
    cur_image_ids = cell(0,1);  %각 변수를 초기화.
    template_size = feature_params.template_size; %외부에서 받아온 feature_params sturcture 내의 변수를 사용의 편의를 위해 재선언.
    cell_size = feature_params.hog_cell_size;
    numcell = template_size/cell_size; %template size에 cell이 몇개 들어가는지 계산.

    threshold = 0.4; %임의로 변경가능!!
    scale = 1.4;  %초기 scale양.임의로 변경가능!!

    while 1 %아래의 if문의 조건이 만족할때까지 계속 반복.
        temp_img = imresize(img,scale); %이미지의 scaling
        if size(temp_img,1) < template_size || size(temp_img,2) < template_size
            break %scaling을 한 image의 x,y크기가 template size 보다 작아지면, 더이상 template window를 sliding 할수 없으므로 종료.
        end
        hog = vl_hog(temp_img, cell_size); %HoG를 계산
        numcell_x = size(hog,2); %계산한 HoG에서 각각의 x,y축에 대해 cell의 갯수를 파악.(sliding을 얼만큼 할것인지 계산하기 위함.)
        numcell_y = size(hog,1);
        for k = 1:numcell_y - numcell + 1 %template window 안에 이미 numcell만큼의 cell이 있으므로 이를 제외한 cell갯수만큼 반복
            for j = 1:numcell_x - numcell + 1 %x,y축 모두 sliding 시킨다.
                cursor = hog(k:k + numcell - 1, j:j + numcell - 1,:); %cursor는 현재 i,j번째 sliding된 window의 hog값을 의미.
                confi = sum(cursor(:) .* w) + b; % SVM training을 통해 얻은 w와 b를 이용해 confidence(score)를 계산.
                if confi > threshold %만약 이 confidence 값이 우리가 정한 기준치를 넘어서면 이를 face로 판별하고 아래와 같이 값을 저장.
                    bbox = [j,k,j + numcell - 1,k + numcell - 1];
                    bbox = bbox * cell_size/scale;
                    cur_x_min = [cur_x_min; bbox(2)];
                    cur_y_min = [cur_y_min; bbox(1)];
                    cur_bboxes = [cur_bboxes; bbox];
                    cur_confidences = [cur_confidences; confi];
                    cur_image_ids = [cur_image_ids; {test_scenes(i).name} ];
                end
            end
        end
        scale = scale -0.1; %scale down
    end