input_root = './images/';
close all;
imnames = dir([input_root '*' 'JPG']);
%Just take the last 25
skyCoordinates = zeros(5,0);
landCoordinates = zeros(5,0);
waterCoordinates = zeros(5,0);
for i=75:length(imnames)
    imorig = imread(strcat(input_root,'/',imnames(i).name));
    imorig = imresize(imorig, .25);
    im = rgb2hsv(imorig);
    [height, width, dim] = size(im);
    figure, imshow(imorig);
    im=im*255;
    hold on;
    legend = sprintf('step 1: Draw line below sky\n step 2: Draw line at horizon');
    %Anything above the sky line is sky
    %Anything below horizon is water
    %Anything above water and below sky is land
    numLines = 0;
    lineSlopes = zeros(2,1);
    lineIntercepts = zeros(2,1);
    while numLines < 2
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
                reset = true;
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
        
        if (reset == false)
            numLines = numLines + 1;
        end
        hold on;
        slope = (y_clicks(2)-y_clicks(1))/(x_clicks(2)-x_clicks(1));
        intercept = y_clicks(1)-slope*x_clicks(1);
        lineSlopes(numLines, 1) = slope;
        lineIntercepts(numLines, 1) = intercept;
        
        h = line([1, width], ...
            [intercept, ...
            width*slope+intercept], ...
            'color', 'r', 'LineWidth', 1);
        %plotHandles = [plotHandles, h];
        pause(0.001);
    end
    %Take two lines and populate coordinates list
    skySlope = lineSlopes(1,1);
    skyIntercept = lineIntercepts(1,1);
    waterSlope = lineSlopes(2,1);
    waterIntercept = lineIntercepts(2,1);
    curSkyCoordinates = zeros(5, width*height);
    curWaterCoordinates = zeros(5, width*height);
    curLandCoordinates = zeros(5, width*height);
    index = 0;
    for col = 1:width
        for row = 1:height
            index = index+1;
            if (row < skyIntercept+skySlope*col)
                curSkyCoordinates(:, index) = [col; row; im(row, col, 1); im(row, col, 2); im(row, col,3)];
            elseif (row > waterIntercept+waterSlope*col)
                curWaterCoordinates(:, index) = [col; row; im(row, col, 1); im(row, col, 2); im(row, col,3)];
            else
                curLandCoordinates(:, index) = [col; row; im(row, col, 1); im(row, col, 2); im(row, col,3)];
            end
        end
    end
    skyCoordinates = [skyCoordinates, curSkyCoordinates(:, curSkyCoordinates(1,:) ~= 0)];
    landCoordinates = [landCoordinates, curLandCoordinates(:, curLandCoordinates(1,:) ~= 0)];
    waterCoordinates = [waterCoordinates, curWaterCoordinates(:, curWaterCoordinates(1,:) ~= 0)];
    close;
end

mSky = mean(skyCoordinates, 2);
coSky = cov(transpose(skyCoordinates));

mLand = mean(landCoordinates, 2);
coLand = cov(transpose(landCoordinates));

mWater = mean(waterCoordinates, 2);
coWater = cov(transpose(waterCoordinates));

