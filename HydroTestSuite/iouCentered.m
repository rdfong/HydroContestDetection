function [iou] = iouCentered(b1, b2)
    w1 = b1(1);
    h1 = b1(2);
    w2 = b2(1);
    h2 = b2(2);
    

    a1 = w1*h1;
    a2 = w2*h2;

    intersection = double(min([w1,w2])*min([h1,h2]));
    union = double((a1+a2)-intersection);
    iou = intersection/union;
end