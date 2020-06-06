close all;
clear;
clc;
%载入图像
TargetImg = imread('修复用图/草地/破损图.jpg');
SourceImg = imread('修复用图/草地/掩膜.jpg');
FullImg = imread('修复用图/草地/原图.jpg');
SourceMask = im2bw(imread('修复用图/草地/mask.jpg'));
%获取裁剪边界
[SrcBoundry,~] = bwboundaries(SourceMask);
%绘制剪切边界
figure,imshow(SourceImg),axis image
hold on
for k = 1:length(SrcBoundry)
    boundary = SrcBoundry{k};
    plot(boundary(:,2),boundary(:,1),'r','LineWidth',2)
end
title('Source image intended area for cutting from');
%确定source图将要粘贴在target途中的具体位置，并产生新的Mask图
 position_in_target = [225,108];
[TargetRows,TargetCols,~] = size(TargetImg);%目标图的尺寸
[FusRow_in_SrcMask,FusCol_in_SrcMask] = find(SourceMask);%查找SourceMask中融合区的坐标
startFus_pos = [min(FusCol_in_SrcMask),min(FusRow_in_SrcMask)];
endFus_pos = [max(FusCol_in_SrcMask),max(FusRow_in_SrcMask)];%找到融合开始点和结束点
Fusion_size = endFus_pos - startFus_pos;
%如果在position_in_target的位置放置Fusion区将超出Target图的范围，则改变position_in_target，以保证Fusion区不会超出Target图的范围。
if (Fusion_size(1) + position_in_target(1) > TargetCols)
    position_in_target(1) = TargetCols - Fusion_size(1);
end

if (Fusion_size(2) + position_in_target(2) > TargetRows)
    position_in_target(2) = TargetRows - Fusion_size(2);
end
%构建一个大小与Target图相等的新的Mask,该Mask图为理想融合位置
MaskTarget = zeros(TargetRows, TargetCols);
MaskTarget(sub2ind([TargetRows, TargetCols], FusRow_in_SrcMask - startFus_pos(2) + position_in_target(2), ...
 FusCol_in_SrcMask - startFus_pos(1) + position_in_target(1))) = 1;
figure, imshow(MaskTarget), axis image
%获取新MaskTarget图的信息
[FusRow_in_NewMask,FusCol_in_NewMask] = find(MaskTarget);%查找MaskTarget中融合区的坐标
Number_Fusion = length(FusRow_in_NewMask);
%对整个源图执行拉普拉斯算子，然后提取RGB三个分量
templt = [0 1 0; 1 -4 1; 0 1 0];
LaplacianSource = imfilter(double(SourceImg), templt, 'replicate');
VR = LaplacianSource(:, :, 1);
VG = LaplacianSource(:, :, 2);
VB = LaplacianSource(:, :, 3);
%然后根据Mask，把上述计算结果贴入TargetImg。
TargetImgR = double(TargetImg(:, :, 1));
TargetImgG = double(TargetImg(:, :, 2));
TargetImgB = double(TargetImg(:, :, 3));

TargetImgR(logical(MaskTarget(:))) = VR(SourceMask(:));
TargetImgG(logical(MaskTarget(:))) = VG(SourceMask(:));
TargetImgB(logical(MaskTarget(:))) = VB(SourceMask(:));

TargetImgNew = cat(3, TargetImgR, TargetImgG, TargetImgB);
figure, imagesc(uint8(TargetImgNew)), axis image, title('Target image with laplacian of source inserted');
%准备工作已经结束，准备进入核心阶段:AX=b
%思路：b列矩阵就是TragetImagNew的值，X就是整个图像，A的构建使用if判断式
%首先构建b列阵
b_r_Max = TargetImgNew(:,:,1)';
b_g_Max = TargetImgNew(:,:,2)';
b_b_Max = TargetImgNew(:,:,3)';
b_r = b_r_Max(:);
b_g = b_g_Max(:);
b_b = b_b_Max(:);
%再使用spalloc函数创建稀疏矩阵A
size_A = length(b_r);
A = spalloc(size_A, size_A, Number_Fusion*5+(size_A-Number_Fusion));
%对A重新赋值
for i=1:size_A
    for j=1:size_A
        if(i==j)
            A(i,j)=1;
        end
    end
end
for i = 1:length(FusRow_in_NewMask)
    PositionOfFusion_inA = (FusRow_in_NewMask(i)-1)*TargetCols+FusCol_in_NewMask(i);
    A(PositionOfFusion_inA,PositionOfFusion_inA) = -4;
    A(PositionOfFusion_inA,PositionOfFusion_inA-1) = 1;
    A(PositionOfFusion_inA,PositionOfFusion_inA+1) = 1;
    A(PositionOfFusion_inA,PositionOfFusion_inA+TargetCols) = 1;
    A(PositionOfFusion_inA,PositionOfFusion_inA-TargetCols) = 1;
end
%双共轭梯度法解AX=b方程
f_r=bicg(A,b_r,1e-6,400);
f_g=bicg(A,b_g,1e-6,400);
f_b=bicg(A,b_b,1e-6,400);
%组成新的融合图片
FusionImage = zeros(TargetRows, TargetCols,3);
for i = 1:TargetRows
    for j = 1:TargetCols
        FusionImage(i,j,1) = f_r((i-1)*TargetCols+j);
        FusionImage(i,j,2) = f_g((i-1)*TargetCols+j);
        FusionImage(i,j,3) = f_b((i-1)*TargetCols+j);
    end
end
ImgFusion = cat(3, FusionImage(:,:,1), FusionImage(:,:,2), FusionImage(:,:,3));
figure, imagesc(uint8(ImgFusion)), title('FusionImage');

Clean_Img=rgb2gray(FullImg);
noise_image=rgb2gray(TargetImg);
Denoising_image=rgb2gray(uint8(ImgFusion));
[PSNR_0,SNR_0]=psnr(uint8(noise_image),uint8(Clean_Img));%计算去噪之前的信噪比
[PSNR_1,SNR_1] = psnr(uint8(Denoising_image),uint8(Clean_Img));  %计算去噪之后的信噪比

