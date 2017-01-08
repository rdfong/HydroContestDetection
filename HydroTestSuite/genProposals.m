input_im = './images/';
ims = dir([input_im '*' 'JPG']);
%Generates proposals and scores them.
%For training, train with every other image starting at the 1st image
%For testing, start at the second
rmdir('./proposals', 's');
mkdir('./proposals');

%for i=1:2:length(ims)
for i=2:2:length(ims)
    %exepath = '../builds/build-MST-Unnamed-Release/MST';
    %exepath = '../builds/build-EMModel-gcc48-Release/EMModel';
    exepath = '../builds/build-MST_GMM_Final-gcc48-Release/MST_GMM_Final';
    cmd = [exepath,' ./images/ ',ims(i).name,' proposals/',' SKIP']
    system(cmd);
end

score;