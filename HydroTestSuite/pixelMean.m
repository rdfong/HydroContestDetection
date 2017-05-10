input_im = './images/';

ims = dir([input_im '*' 'JPG']);

runningSum = [0,0,0];
for i=1:length(ims)
    im = imread(strcat(input_im,'/',ims(i).name));
    runningSum = runningSum + reshape(mean(mean(im)), 1,3);
end
runningSum = runningSum/length(ims)