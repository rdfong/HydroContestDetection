input_proposal = './proposals/';
input_annotation = './annotations/';

outID = fopen( strcat(datestr(datetime('now')),'results.txt'), 'wt' );

proposals = dir([input_proposal '*' 'txt']);

recallResults = [];
precisionResults = [];
fscoreResults = [];

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
            proposalClasses = [proposalClasses; tline];
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
            annotationClasses = [annotationClasses; tline];
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
    totalProposalArea = 0;
    proposalTotal = 0;
    proposalMatches = 0;
    for (i = 1:size(proposalClasses, 1))
        proposalBoxCell = proposalBoxes(i,:);
        proposalBox = proposalBoxCell{1};
        pX = proposalBox(1);
        pY = proposalBox(2);
        pWidth = proposalBox(3);
        pHeight = proposalBox(4);
        totalProposalArea = totalProposalArea + pWidth*pHeight;
        proposalMatrix = zeros(pHeight, pWidth);
        for (j = 1:size(annotationClasses, 1))
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
        end
        proposalTotal = proposalTotal + sum(sum(proposalMatrix));
    end
    
    if totalProposalArea == 0
        precision = 1;
    else
        precision = double(proposalTotal)/double(totalProposalArea);
    end
    
    totalAnnotationArea = 0;
    annotationTotal = 0;
    for (i = 1: size(annotationClasses, 1))
        annotationBoxCell = annotationBoxes(i,:);
        annotationBox = annotationBoxCell{1};
        aX = annotationBox(1);
        aY = annotationBox(2);
        aWidth = annotationBox(3);
        aHeight = annotationBox(4);
        totalAnnotationArea = totalAnnotationArea + aWidth*aHeight;
        annotationMatrix = zeros(aHeight, aWidth);
        for (j = 1:size(proposalClasses, 1))
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
        end
        annotationTotal = annotationTotal + sum(sum(annotationMatrix));
    end
    
    if totalAnnotationArea == 0
        recall = 1;
    else
        recall = double(annotationTotal)/double(totalAnnotationArea);
    end
    
    fscore = 2*precision*recall/(precision+recall);
    
    fprintf(outID, proposals(p).name);
    message = sprintf('\nRecall: %f\nPrecision: %f\nFScore: %f\n', ...
        recall, precision, fscore);
    fprintf(outID, message);
    
    precisionResults = [precisionResults, precision];
    recallResults = [recallResults, recall];
    fscoreResults = [fscoreResults, fscore];
end

message = sprintf('\n\nTotal Recall: %f\nTotal Precision: %f\nTotal FScore: %f\n', ...
    mean(precisionResults), mean(recallResults), mean(fscoreResults));
fprintf(outID, message);
fclose(outID);
%Write score to file (saveas dialog)
%show indivual scores for each file and a total final score