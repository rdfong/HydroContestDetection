input_im = './images/';
ims = dir([input_im '*' 'JPG']);
for i=1:length(ims)
    %exepath = '../builds/build-MST-Unnamed-Release/MST';
    exepath = '../builds/build-EMModel-gcc48-Release/EMModel';
    cmd = [exepath,' ./images/ ',ims(i).name,' proposals/',' SKIP']
    system(cmd);
end

score;