a_dir = './annotations/';
im_dir = './images/';
a_files = dir([a_dir '*' 'txt']);
im_files = dir([im_dir '*' 'JPG']);
for i=1:length(a_files)
    anno_id = fopen(strcat(a_dir,'/',a_files(i).name));
    im = imread(strcat(im_dir,'/',im_files(i).name));
    
    voc_file = strcat(a_dir,'VOC_annotations/',a_files(i).name);
    fileID = fopen(voc_file, 'w');

    [height, width, dim] = size(im);
    
    fprintf(fileID, '<annotation>\n');
    fprintf(fileID, '    <folder>VOC2007</folder>\n');
    fprintf(fileID,['    <filename>' im_files(i).name '</filename>\n']);
	fprintf(fileID, '    <size>\n');
	fprintf(fileID,['        <width>' int2str(width) '</width>\n']);
    fprintf(fileID,['        <height>' int2str(height) '</height>\n']);
	fprintf(fileID,['        <depth>' int2str(dim) '</depth>\n']);
    fprintf(fileID, '    </size>\n');
    
    class = fgetl(anno_id);
    while ischar(class)
        bbox_line = fgetl(anno_id);
        [b_x, b_y, b_w, b_h] = strread(bbox_line, '%d %d %d %d');
        fprintf(fileID, '    <object>\n');
        fprintf(fileID,['        <name>' class '</name>\n']);
        fprintf(fileID,	'        <pose>Unspecified</pose>\n');
		fprintf(fileID, '        <truncated>0</truncated>\n');
		fprintf(fileID, '        <difficult>0</difficult>\n');
        fprintf(fileID,['        <xmin>' int2str(b_x) '</xmin>\n']);
        fprintf(fileID,['        <xmax>' int2str(b_x+b_w-1) '</xmax>\n']);
        fprintf(fileID,['        <ymin>' int2str(b_y) '</ymin>\n']);
        fprintf(fileID,['        <ymax>' int2str(b_y+b_h-1) '</ymax>\n']);
        fprintf(fileID, '    </object>\n');
        class = fgetl(anno_id);
    end
    
    fprintf(fileID, '</annotation>\n');
    
    [pathstr, name, ext] = fileparts(voc_file);
    movefile(voc_file, fullfile(pathstr, [name '.xml']));
    
end