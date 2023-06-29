% Approximate Median Filter background model for moving object segmentation.
%���ý�����ֵ�˲�����ģ�Ͳο�ͼ��ʵ���˶�Ŀ��ָ�
clear all; close all;
% Construct a videoreader class to read a avi file, first the 'ygrmz.avi' , 
% then the ��highwayII_raw.avi'.
videoObj = VideoReader('ygrmz.avi');
numFrames =videoObj.NumberOfFrames;
%Get the speed of the AVI movie in frames per second (fps)
FPS = videoObj.FrameRate;
% Read the first frame in the video sequence as the initial value 
newframe = read(videoObj, 1);
fmed = double(newframe); %��������Ƶ���ݸ�ʽת��Ϊdouble��
% Get the height, width, and number of color components of the frame
[height, width, numColor] = size(newframe);
% Assign a value to the threshold
Threh = 50;%�趨��ֵ
beta = 0.6;
fg = false(height, width);
%�������νṹԪ�أ����ڶԷָ�����̬ѧ�˲�
se = strel('square',3);
% To avoid consuming too much memories, read only a one frame each time.
for n = 1:numFrames   
    newframe = read(videoObj, n);
   	% Calculate the differrence image between the new frame and fmed        
    Idiff = double(newframe) - fmed;
    % Update the median of each pixel value
    pixInc = find(Idiff > 0);
    fmed(pixInc) = fmed(pixInc) + beta;
    pixDec = find(Idiff < 0);
    fmed(pixDec) = fmed(pixDec) - beta;
    % Motion segment, detection moving object by threholding Idiff 
    fg = abs(Idiff) >Threh;
    if ( numColor == 3)  % color image
       	fg = fg(:, :, 1) | fg(:, :, 2) | fg(:, :, 3);  
    end
    %�Էָ���������̬ѧ�˲�
    fg2 = imopen(fg,se);%������
    fg2 = imclose(fg2,se);%�ղ���
     A=regionprops(fg2,'basic'); %��ȡ��ͨ��
    [C,area_index]=max([A.Area]); %��ȡ�����������Ϊ��
    figure(1);
   	subplot(2,2,1), imshow(newframe);
    title(strcat('Current Image, No. ', int2str(n))); %��ǰͼ��
    subplot(2,2,2), imshow(fg);
    title('Segmented result using Approximate Median Filter');%������ֵ�˲�
    subplot(2,2,3), imshow(fg2);
     title('Segmented result using morphological filter');%��̬ѧ�˲�
     subplot(2,2,4), imshow(newframe);
     title(strcat('Current Image, No. ', int2str(n))); %��ǰͼ��
       hold on;
         if(C~=0)
         rectangle('Position',[A(area_index).BoundingBox],'LineWidth',2,'LineStyle','-','EdgeColor','r');
         end
         title('���˵�׷��');
         pause(0.01);       
    
end
%----------------------------------------------------------------------
