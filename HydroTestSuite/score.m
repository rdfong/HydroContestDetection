input_proposal = './proposals/';
input_annotation = './annotations/';

outID = fopen( strcat(datestr(datetime('now')),'results.txt'), 'wt' );

proposals = dir([input_proposal '*' 'txt']);

recallResults = [];
precisionResults = [];
fscoreResults = [];
mboResults = [];

for p=1:length(proposals)
    proposalClasses = [];
    proposalBoxes = [];
    annotationBoxes = [];
    annotationClasses = [];
    
    %Read proposals
    proposalID = fopen(strcat(input_proposal,proposals(p).name));
    tline = fgetl(proposalID);
    startBox = true;
    while ischar(tline)
        if startBox
            %proposalClasses = [proposalClasses; tline];
        else
            proposalBoxes = [proposalBoxes; textscan(tline,'%d')];
        end
        tline = fgetl(proposalID);
        startBox = ~startBox;
    end
    fclose(proposalID);
    
    %Read in annotations
    annotationID = fopen(strcat(input_annotation,proposals(p).name));
    tline = fgetl(annotationID);
        startBox = true;
    while ischar(tline)
        if startBox
            %annotationClasses = [annotationClasses; tline];
        else
            annotationBoxes = [annotationBoxes; textscan(tline,'%d')];
        end
        tline = fgetl(annotationID);
        startBox = ~startBox;
    end
    fclose(annotationID);
    
    %Create F score for both boxes and classes for each image
    
    %Boxes that have a significant overlap with the ground truth to be
    %considered for a classification score. All others are misses.
    matchingBoxIds = [];
    
    
    %precision of boxes
    proposalTotal = 0;
    for (i = 1:size(proposalBoxes, 1))
        proposalBoxCell = proposalBoxes(i,:);
        proposalBox = proposalBoxCell{1};
        pX = proposalBox(1);
        pY = proposalBox(2);
        pWidth = proposalBox(3);
        pHeight = proposalBox(4);
        maxOverlap = 0;
        for (j = 1:size(annotationBoxes, 1))
            proposalMatrix = zeros(pHeight, pWidth);
            annotationBoxCell = annotationBoxes(j,:);
            annotationBox = annotationBoxCell{1};
            aX = annotationBox(1);
            aY = annotationBox(2);
            aWidth = annotationBox(3);
            aHeight = annotationBox(4);
            %How much of current proposalBox overlaps with annotation
            if pX >= aX+aWidth || pY >= aY+aHeight || pX+pWidth <= aX || pY+pHeight <= aY
                %no overlap
                continue;
            else
                new_aX = max(pX, aX)-pX+1;
                new_aY = max(pY, aY)-pY+1;
                new_aX2 = min(pX+pWidth, aX+aWidth)-pX;
                new_aY2 = min(pY+pHeight, aY+aHeight)-pY;
            end
            proposalMatrix(new_aY:new_aY2, new_aX:new_aX2) = 1;
            intersection = double(sum(sum(proposalMatrix)));
            union = double((pWidth*pHeight+aWidth*aHeight)-intersection);
            proposalArea = pWidth*pHeight;
            %only consider anything with an overlap score of at least 0.5
            if (intersection/proposalArea > 0.5)
                maxOverlap = max(intersection/proposalArea, maxOverlap);
            end
        end
        proposalTotal = proposalTotal + maxOverlap;
    end
    
    %This returns average precision for the image, but we may want to do
    %precision per box isntead of per image?
    if size(proposalBoxes, 1) == 0
        precision = 1;
    else
        precision = double(proposalTotal)/size(proposalBoxes, 1);
    end
    
    annotationTotal = 0;
    mboTotal = 0;
    for (i = 1:size(annotationBoxes, 1))
        annotationBoxCell = annotationBoxes(i,:);
        annotationBox = annotationBoxCell{1};
        aX = annotationBox(1);
        aY = annotationBox(2);
        aWidth = annotationBox(3);
        aHeight = annotationBox(4);
        maxOverlap = 0;
        maxOverlapForMBO = 0;
        for (j = 1:size(proposalBoxes, 1))
            annotationMatrix = zeros(aHeight, aWidth);
            proposalBoxCell = proposalBoxes(j,:);
            proposalBox = proposalBoxCell{1};
            pX = proposalBox(1);
            pY = proposalBox(2);
            pWidth = proposalBox(3);
            pHeight = proposalBox(4);
            if aX >= pX+pWidth || aY >= pY+pHeight || aX+aWidth <= pX || aY+aHeight <= pY
                %no overlap
                continue;
            else
                new_pX = max(aX, pX)-aX+1;
                new_pY = max(aY, pY)-aY+1;
                new_pX2 = min(aX+aWidth, pX+pWidth)-aX;
                new_pY2 = min(aY+aHeight, pY+pHeight)-aY;
            end
            annotationMatrix(new_pY:new_pY2, new_pX:new_pX2) = 1;
            intersection = double(sum(sum(annotationMatrix)));
            annotationArea = aWidth*aHeight;
            union = double((pWidth*pHeight+aWidth*aHeight)-intersection);
            %only consider anything with an overlap score of at least 0.5
            if (intersection/annotationArea > 0.5)
                maxOverlap = max(intersection/annotationArea, maxOverlap);
            end
            if (intersection/union > 0.5)
                maxOverlapForMBO = max(intersection/union, maxOverlapForMBO);
            end
        end
        annotationTotal = annotationTotal + maxOverlap;
        mboTotal = mboTotal + maxOverlapForMBO;
    end
    
    if size(annotationBoxes, 1) == 0
        mbo = 1;
        recall = 1;
    else
        mbo = double(mboTotal)/size(annotationBoxes, 1);
        recall = double(annotationTotal)/size(annotationBoxes, 1);
    end
    
    fscore = 0;
    if (precision+recall > 0)
        fscore = 2*precision*recall/(precision+recall);
    end
    
    fprintf(outID, proposals(p).name);
    message = sprintf('\nMBO: %f\nRecall: %f\nPrecision: %f\nFScore: %f\n', ...
        mbo, recall, precision, fscore);
    fprintf(outID, message);
    
    precisionResults = [precisionResults, precision];
    recallResults = [recallResults, recall];
    fscoreResults = [fscoreResults, fscore];
    mboResults = [mboResults, mbo];
end

%Find mean of results over all images
message = sprintf('\n\nMBO: %f\nTotal Recall: %f\nTotal Precision: %f\nTotal FScore: %f\n', ...
    mean(mboResults), mean(recallResults), mean(precisionResults), mean(fscoreResults));
fprintf(outID, message);
fclose(outID);
%Write score to file (saveas dialog)
%show indivual scores for each file and a total final score