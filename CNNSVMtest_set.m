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