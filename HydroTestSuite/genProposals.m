input_im = './images/';
ims = dir([input_im '*' 'JPG']);
for i=2:2:length(ims)
    %exepath = '../builds/build-MST-Unnamed-Release/MST';
    %exepath = '../builds/build-EMModel-gcc48-Release/EMModel';
    exepath = '../builds/build-MST_GMM_Final-gcc48-Release/MST_GMM_Final';
    cmd = [exepath,' ./images/ ',ims(i).name,' proposals/',' SKIP']
    system(cmd);
end

score;