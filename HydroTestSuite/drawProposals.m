close all;
image_indices =  {'03'; '15'; '21'; '57'; '65'; '68'; '99'; '79'; '42'; '63'; '107'; '115'; '120'; '135'; '145'};
confidence = 0.05;

p_dir = './proposals/';
im_dir = './images/';

for i=1:length(image_indices)
    imstr = strcat(im_dir,image_indices(i),'.JPG');
    im = imread(imstr{1});
    
    p_file = strcat(p_dir,image_indices(i),'.JPG.txt');
    fileID = fopen(p_file{1}, 'r');
    %now read class, confidence and box
    figure, imshow(im);
    title(strcat(image_indices(i),'.JPG'), 'FontSize', 16)
    
    tline = fgetl(fileID);
    while ischar(tline)
        proposalClass = tline;
        
        tline = fgetl(fileID);
        confProb = str2double(tline);
        
        tline = fgetl(fileID);
        if confProb >= confidence
            box = textscan(tline,'%d');
            box = box{1}
            hold on;
            rectangle('Position',[box(1),box(2),box(3),box(4)], 'EdgeColor', 'r', 'LineWidth', 2.0);
        end
        
        tline = fgetl(fileID);
    end

    fclose(fileID);
    pause(0.5);
end