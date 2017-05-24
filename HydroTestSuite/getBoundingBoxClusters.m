input_annotation = './annotations/';
K = [2:20];
%K=[5]
annotations = dir([input_annotation '*' 'txt']);

gridSize = 13.0;
imageWidth = 500.0;
imageHeight = 333.0;

gridWidth = imageWidth/gridSize;
gridHeight = imageHeight/gridSize;

boxDimensions = [];
for i = 1:length(annotations)
    annoID = fopen(strcat(input_annotation,annotations(i).name));
    tline = fgetl(annoID);
    startBox = true;
    while ischar(tline)
        if ~startBox
            annoBoxes = textscan(tline,'%d');
            annoBoxes = annoBoxes{1};
            boxDimensions = [boxDimensions;annoBoxes(3:4)'];
        end
        tline = fgetl(annoID);
        startBox = ~startBox;
    end
    fclose(annoID);
end

boxDimensions = double(boxDimensions);
boxDimensions(:,1) = boxDimensions(:,1)/gridWidth;
boxDimensions(:,2) = boxDimensions(:,2)/gridHeight;

errorVec = [];
for k = K
    [C,idx,error] = customKMeans(boxDimensions, k, 1000, 1e-6);
    errorVec = [errorVec, error];
end

figure, plot(K, errorVec, 'o');
