
%% Step 1:  Read Images
% Read the reference image containing the object of interest.
boxImage = rgb2gray(imread('23-1.tif'));
figure;
imshow(boxImage);
title('image of box');

%%
% Read the target image containing a cluttered scene.
sceneImage = rgb2gray(imread('40-1.tif'));
figure;
imshow(sceneImage);
title('Image of a Cluttered Scene');

%% Step 2: Detect Feature Points
% Detect feature points in both images.
boxPoints = detectSURFFeatures(boxImage);
scenePoints = detectSURFFeatures(sceneImage);

boxBriskPoints = detectBRISKFeatures(boxImage);
sceneBriskPoints = detectBRISKFeatures(sceneImage);


%%
% Visualize the Strongest feature points found in the reference image.
figure;
imshow(boxImage);
title('100 Strongest Feature Points from Box Image');
hold on;
plot(selectStrongest(boxPoints, 100));

figure;
imshow(boxImage);
title('100 Strongest Feature Points from box Image');
hold on;
plot(selectStrongest(boxBriskPoints, 100));


%%
% Visualize the strongest feature points found in the target image.
figure;
imshow(sceneImage);
title('300 Strongest Feature Points from Scene Image');
hold on;
plot(selectStrongest(scenePoints, 300));

%% Step 3: Extract Feature Descriptors
% Extract feature descriptors at the interest points in both images.
[boxFeatures, boxPoints] = extractFeatures(boxImage, boxPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

%% Step 4: Find Putative Point Matches
% Match the features using their descriptors.
boxPairs = matchFeatures(boxFeatures, sceneFeatures);

%%
% Display putatively matched features.
matchedBoxPoints = boxPoints(boxPairs(:, 1), :);
matchedScenePoints = scenePoints(boxPairs(:, 2), :);
figure;
showMatchedFeatures(boxImage, sceneImage, matchedBoxPoints, ...
    matchedScenePoints, 'montage');
title('Putatively Matched Points (Including Outliers)');