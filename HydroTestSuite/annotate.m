input_root = './images/';
output_root = './annotations/';
numClasses = 5;
close all;

if ~exist(output_root, 'dir')
    mkdir(output_root);
end

imnames = dir([input_root '*' 'JPG']);
for i=1:length(imnames)
    im = imread(strcat(input_root,'/',imnames(i).name));
    [height, width, dim] = size(im);
    figure, imshow(im);
    hold on;
    legend = sprintf('1-Boat\n2-Buoy\n3-Person\n4-Bird\n5-Other');
    text(0,30,legend,'Color',[1,0,0],'FontSize',10);
    
    outFile = strcat(output_root,'/',imnames(i).name, '.txt');
    fileID = fopen(outFile, 'w');
   
    while 1
        exitCondition = 0;
        numClicks = 0;
        x_clicks = [];
        y_clicks = [];
        plotHandles = [];
        while (numClicks < 2) 
            [ClickedX, ClickedY, PressedKey] = ginput(1);
            pause(0.001);
            if PressedKey == 32
                exitCondition = 1;
                break;
            elseif PressedKey == 8
                for p = 1:length(plotHandles)
                    delete(plotHandles(p));
                end
                exitCondition = 2;
                break;
            elseif PressedKey ~= 1
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
        if (exitCondition == 1)
            break;
        elseif exitCondition == 2
            continue;
        end
        minX = min(x_clicks);
        minY = min(y_clicks);
        maxX = max(x_clicks);
        maxY = max(y_clicks);
        
        if (maxX-minX == 0 || maxY-minY == 0)
            for p = 1:length(plotHandles)
                delete(plotHandles(p));
            end
            continue;
        end

        hold on;
        h = rectangle('Position',[minX,minY,maxX-minX,maxY-minY], 'EdgeColor', 'r');
        plotHandles = [plotHandles, h];
        % fprintf(fileID,'\n','','exp(x)');
        %1-boat
        %2-buoy
        %3-person
        %4-bird
        %5-other
        hold on;
        classSelected = 0;
        while (classSelected == 0)
            [ClickedX, ClickedY, PressedKey] = ginput(1);
            pause(0.001);
            if PressedKey >= 49 && PressedKey < 49+numClasses
                classSelected = PressedKey - 48;
                txt = num2str(classSelected);
                hold on;
                text(minX,maxY+5,txt,'Color',[1,0,0], 'FontSize', 15);
            elseif PressedKey == 8
                for p = 1:length(plotHandles)
                    delete(plotHandles(p));
                end
                break;
            end
        end
        pause(0.001);
        
        className = '';
        switch classSelected
            case 1
                className = 'boat';
            case 2
                className = 'buoy';
            case 3
                className = 'person';
            case 4
                className = 'bird';
            case 5
                className = 'other';
        end

        %Write results to file, in 0 index form
        if classSelected > 0
            fprintf(fileID, strcat(className, '\n'));
            fprintf(fileID, '%d %d %d %d\n', uint32(minX-1), uint32(minY-1), uint32(maxX-minX), uint32(maxY-minY));
        end
    end
    
    saveas(gcf, strcat(output_root,'/',imnames(i).name, '.jpg'));
    close;
    fclose(fileID);
end