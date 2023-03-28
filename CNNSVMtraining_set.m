clc
clear all
close all

%% Programmatic Transfer Learning Using Convolution neural networks-Support vector machine
%LoadData
% unzip('MerchData.zip');
imds = imageDatastore('SingleCTCData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%% Input layer random forest(227-pixel size)
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,100);

I = imtile(imds, 'Frames', idx);

figure
imshow(I)
%Load Pretrained Network(based on model loading)
net = squeezenet;

analyzeNetwork(net)
inputSize = net.Layers(1).InputSize

% Extract Image Features (length scale&shape factor&IoU&RGBindex)
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'pool10';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

%Fit Image Classifier (loda SVM classifier)
mdl = fitcecoc(featuresTrain,YTrain);

%Classify Test Images
YPred = predict(mdl,featuresTest);
% idx = [1 5 10 15 20 25 30 35 40 45 50 60 65 70 74]; 
%  idx = [1 13  25 37 49 61 73 85 97 109 121 133 155 167 179 191];
idx = randi([1,max(size(imdsTest.Files))],1,16);


figure
for i = 1:numel(idx)
    subplot(4,4,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    
    imshow(I)
    title(label)
end

accuracy = mean(YPred == YTest)


% rng(0,'twister');
% a = 1;
% b = 73;
% idx = (b-a).*rand(16,1) + a;
% idx = randi([1,677],1,16);
idx= randi([1,max(size(imdsTest.Files))],1,16);

figure
for i = 1:numel(idx)
    subplot(4,4,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    
    imshow(I)
    title(label)
end

accuracy = mean(YPred == YTest)

% data analysis step
classificationLearner



