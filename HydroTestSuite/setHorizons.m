input_root = './images/';
output_root = './annotations/'

close all;
imnames = dir([input_root '*' 'JPG']);
%Just take the last 25
skyCoordinates = zeros(5,0);
landCoordinates = zeros(5,0);
waterCoordinates = zeros(5,0);
for i=1:length(imnames)
    im = imread(strcat(input_root,'/',imnames(i).name));
    im = imresize(im, .25);
    [height, width, dim] = size(im);
    figure, imshow(im);
    hold on;
    %Anything above the sky line is sky
    %Anything below horizon is water
    %Anything above water and below sky is land
    numLines = 0;
    numClicks = 0;
    x_clicks = [];
    y_clicks = [];
    plotHandles = [];
    while (numClicks < 2) 
        [ClickedX, ClickedY, PressedKey] = ginput(1);
        pause(0.001);
        reset = false;
        if PressedKey == 8
            for p = 1:length(plotHandles)
                delete(plotHandles(p));
            end
            numClicks = 0;
            x_clicks = [];
            y_clicks = [];
            continue;
        end

        hold on;
        ClickedX = max(min(ClickedX, width), 1);
        ClickedY = max(min(ClickedY, height), 1);
        h = plot(ClickedX,ClickedY,'r+','MarkerSize',10);
        plotHandles = [plotHandles, h];
        x_clicks = [x_clicks, ClickedX];
        y_clicks = [y_clicks, ClickedY];
        numClicks = numClicks + 1;
    end
    
    hold on;
    slope = (y_clicks(2)-y_clicks(1))/(x_clicks(2)-x_clicks(1));
    intercept = y_clicks(1)-slope*x_clicks(1);

    h = line([1, width], ...
        [intercept, ...
        width*slope+intercept], ...
        'color', 'r', 'LineWidth', 1);
    %now output hydrotestsuite images folder
    outFile = strcat(output_root,'/',imnames(i).name, '_horizon.txt');
    fileID = fopen(outFile, 'w');
    fprintf(fileID, '%d %d %d %d', slope, intercept, width, height); 
    pause(1.0);
    close;
end


