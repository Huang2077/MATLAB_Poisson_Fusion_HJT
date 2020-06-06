close all;
clear;
clc;
%����ͼ��
TargetImg = imread('�޸���ͼ/�ݵ�/����ͼ.jpg');
SourceImg = imread('�޸���ͼ/�ݵ�/��Ĥ.jpg');
FullImg = imread('�޸���ͼ/�ݵ�/ԭͼ.jpg');
SourceMask = im2bw(imread('�޸���ͼ/�ݵ�/mask.jpg'));
%��ȡ�ü��߽�
[SrcBoundry,~] = bwboundaries(SourceMask);
%���Ƽ��б߽�
figure,imshow(SourceImg),axis image
hold on
for k = 1:length(SrcBoundry)
    boundary = SrcBoundry{k};
    plot(boundary(:,2),boundary(:,1),'r','LineWidth',2)
end
title('Source image intended area for cutting from');
%ȷ��sourceͼ��Ҫճ����target;�еľ���λ�ã��������µ�Maskͼ
 position_in_target = [225,108];
[TargetRows,TargetCols,~] = size(TargetImg);%Ŀ��ͼ�ĳߴ�
[FusRow_in_SrcMask,FusCol_in_SrcMask] = find(SourceMask);%����SourceMask���ں���������
startFus_pos = [min(FusCol_in_SrcMask),min(FusRow_in_SrcMask)];
endFus_pos = [max(FusCol_in_SrcMask),max(FusRow_in_SrcMask)];%�ҵ��ںϿ�ʼ��ͽ�����
Fusion_size = endFus_pos - startFus_pos;
%�����position_in_target��λ�÷���Fusion��������Targetͼ�ķ�Χ����ı�position_in_target���Ա�֤Fusion�����ᳬ��Targetͼ�ķ�Χ��
if (Fusion_size(1) + position_in_target(1) > TargetCols)
    position_in_target(1) = TargetCols - Fusion_size(1);
end

if (Fusion_size(2) + position_in_target(2) > TargetRows)
    position_in_target(2) = TargetRows - Fusion_size(2);
end
%����һ����С��Targetͼ��ȵ��µ�Mask,��MaskͼΪ�����ں�λ��
MaskTarget = zeros(TargetRows, TargetCols);
MaskTarget(sub2ind([TargetRows, TargetCols], FusRow_in_SrcMask - startFus_pos(2) + position_in_target(2), ...
 FusCol_in_SrcMask - startFus_pos(1) + position_in_target(1))) = 1;
figure, imshow(MaskTarget), axis image
%��ȡ��MaskTargetͼ����Ϣ
[FusRow_in_NewMask,FusCol_in_NewMask] = find(MaskTarget);%����MaskTarget���ں���������
Number_Fusion = length(FusRow_in_NewMask);
%������Դͼִ��������˹���ӣ�Ȼ����ȡRGB��������
templt = [0 1 0; 1 -4 1; 0 1 0];
LaplacianSource = imfilter(double(SourceImg), templt, 'replicate');
VR = LaplacianSource(:, :, 1);
VG = LaplacianSource(:, :, 2);
VB = LaplacianSource(:, :, 3);
%Ȼ�����Mask������������������TargetImg��
TargetImgR = double(TargetImg(:, :, 1));
TargetImgG = double(TargetImg(:, :, 2));
TargetImgB = double(TargetImg(:, :, 3));

TargetImgR(logical(MaskTarget(:))) = VR(SourceMask(:));
TargetImgG(logical(MaskTarget(:))) = VG(SourceMask(:));
TargetImgB(logical(MaskTarget(:))) = VB(SourceMask(:));

TargetImgNew = cat(3, TargetImgR, TargetImgG, TargetImgB);
figure, imagesc(uint8(TargetImgNew)), axis image, title('Target image with laplacian of source inserted');
%׼�������Ѿ�������׼��������Ľ׶�:AX=b
%˼·��b�о������TragetImagNew��ֵ��X��������ͼ��A�Ĺ���ʹ��if�ж�ʽ
%���ȹ���b����
b_r_Max = TargetImgNew(:,:,1)';
b_g_Max = TargetImgNew(:,:,2)';
b_b_Max = TargetImgNew(:,:,3)';
b_r = b_r_Max(:);
b_g = b_g_Max(:);
b_b = b_b_Max(:);
%��ʹ��spalloc��������ϡ�����A
size_A = length(b_r);
A = spalloc(size_A, size_A, Number_Fusion*5+(size_A-Number_Fusion));
%��A���¸�ֵ
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
%˫�����ݶȷ���AX=b����
f_r=bicg(A,b_r,1e-6,400);
f_g=bicg(A,b_g,1e-6,400);
f_b=bicg(A,b_b,1e-6,400);
%����µ��ں�ͼƬ
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
[PSNR_0,SNR_0]=psnr(uint8(noise_image),uint8(Clean_Img));%����ȥ��֮ǰ�������
[PSNR_1,SNR_1] = psnr(uint8(Denoising_image),uint8(Clean_Img));  %����ȥ��֮��������

