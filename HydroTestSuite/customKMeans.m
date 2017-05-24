function [C, I,FinalError] = customKMeans(X, K, maxIter, TOL)
FinalError = 0;
% number of vectors in X
[vectors_num, dim] = size(X);

% compute a random permutation of all input vectors
R = randperm(vectors_num);

% construct indicator matrix (each entry corresponds to the cluster
% of each point in X)
I = zeros(vectors_num, 1);

% construct centers matrix
C = zeros(K, dim);

% take the first K points in the random permutation as the center sead
for k=1:K
    C(k,:) = X(R(k),:);
end

iter = 0;
while 1
    % find closest point
    for n=1:vectors_num
        % find closest center to current input point
        minIdx = 1;
        minVal = 1.0-iouCentered(X(n,:),C(minIdx,:));
        for j=1:K
            dist = 1.0-iouCentered(C(j,:),X(n,:));
            if dist < minVal
                minIdx = j;
                minVal = dist;
            end
        end
        
        % assign point to the cluster center
        I(n) = minIdx;
    end
    
    % compute centers
    for k=1:K
        if (length(find(I == k)))
            C(k, :) = sum(X(find(I == k), :));
            C(k, :) = C(k, :) / length(find(I == k));
        else
            C(k, :) = X(R(randi(length(R))),:);
        end
    end
 
    % compute RSS error
    iou_error = 0;
    for idx=1:vectors_num
        iou_error = iou_error + (1.0-iouCentered(X(idx, :), C(I(idx),:)));
    end
    
    iou_error = iou_error / vectors_num;
    % increment iteration
    iter = iter + 1;
    
    % check stopping criteria
    if 1/iou_error < TOL
        FinalError = iou_error;
        break;
    end
    
    if iter > maxIter
        FinalError = iou_error;
        iter = iter - 1;
        break;
    end
end

disp(['k-means took ' int2str(iter) ' steps to converge']);