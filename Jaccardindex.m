A = imread('4-15-2.tif');
I = im2gray(A);
figure
imshow(I)
title('Original Image')

mask = zeros(size(I));
mask(25:end-25,25:end-25) = 1;
imshow(mask)
title('Inital Contour Location')
BW = activecontour(I, mask, 500);

imshow(BW)
title('Segmented Image, 100 Iterations')

%BW = activecontour(I,mask,500);
%imshow(BW)
%title('Segmented Image, 500 Iterations')

BW_groundTruth = imread('40GT-1.png');

similarity = jaccard(BW, BW_groundTruth);

figure
imshowpair(BW, BW_groundTruth)
title(['Jaccard Index = ' num2str(similarity)])