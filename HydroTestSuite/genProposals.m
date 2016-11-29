input_im = './images/';
ims = dir([input_im '*' 'JPG']);
for i=1:length(ims)
    cmd = ['../builds/build-MST-Unnamed-Release/MST',' ./images/ ',ims(i).name,' proposals/',' SKIP']
    system(cmd);
end

score;