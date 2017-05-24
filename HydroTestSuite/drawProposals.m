close all;
p_dir = ''
if strcmp(method,'YOLO')
    p_dir = './proposals/yolo/images/';
elseif strcmp(method, 'RCNN')
    p_dir = './proposals/rcnn/images/';
end

if strcmp(display,'NO')
    display = 0
end

image_indices =  {'00';'03';'06';'09';'12';'15';'18';...
'21';'24';'27';'30';'33';'36';'39';'42';'45';'48';'51';'54';...
'57';'60';'63';'66';'69';'72';'75';'78';'81';'84';'87';'90';...
'93';'96';'99';'102';'105';'108';'111';'114';'117';'120';...
'123';'126';'129';'132';'135';'138';'141';'144';'147'};
confidence = 0.05;

im_dir = './images/';

for i=1:length(image_indices)
    imstr = strcat(im_dir,image_indices(i),'.JPG');
    im = imread(imstr{1});
    
    p_file = strcat(p_dir,image_indices(i),'.JPG.txt');
    fileID = fopen(p_file{1}, 'r');
    %now read class, confidence and box
    h = figure;
    if (display == 0)
        set(h, 'Visible', 'off');
    end
    imshow(im);
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
    if ~display
       saveas(h, strcat(p_dir,image_indices(i),'.jpg'))
    end
    fclose(fileID);
    pause(0.5);
end