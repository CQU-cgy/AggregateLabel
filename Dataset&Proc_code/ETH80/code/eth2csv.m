clear;clc; rng(0);
load('Definition_pkg');
%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
%数据集、预处理方法、编码方法选择
PRE_PROC = 'Gray';                   % 预处理方法，可选：'Gray'或'Dog'
CODING_METHOD = 'Temporal_Coding';  % 编码方法，可选：Temporal_Coding', 'Rate_Coding'或'ISI_Coding'
TRAIN_RATIO = 0.6;                  % 训练集所占比例

%Resize参数
IF_RESIZE = 1;                      % 是否做resize处理，1--True, 0--False
RESIZE_SIZE = 28;                   % resize后的图像大小

%高斯差分滤波（Dog）参数
SIGMA1 = 1; SIGMA2 = 3; WINDOW = 15;

%----------------------------------------------------------------------参数配置区-------------------------------------------------------------------%
IMG_DIR = '..\Original_png\';


all_imgs_2d = zeros(ETH80_IMG, TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));
all_labels = zeros(ETH80_IMG, 1);
all_imgs_2d_rand = zeros(ETH80_IMG, TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));
all_labels_rand = zeros(ETH80_IMG, 1);
train_imgs_2d = zeros(round(ETH80_IMG * TRAIN_RATIO), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));
test_imgs_2d = zeros(round(ETH80_IMG * (1 - TRAIN_RATIO)), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE), TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE));
train_labels = zeros(round(ETH80_IMG * TRAIN_RATIO), 1);
test_labels = zeros(round(ETH80_IMG * (1 - TRAIN_RATIO)), 1);

%读入每张图像，并手动生成标签
img_cnt = 0;
for i = 1 : ETH80_NCLASS %8
    for j = 1 : ETH80_NFOLDER_PER_CLASS %10
        img_list = dir(strcat(IMG_DIR, num2str(i), '\', num2str(j), '\', '*.png')); %获取该目录下所有png格式图像信息
        for n = 1 : ETH80_NIMG_PER_FOLDER %41
            img_name = img_list(n).name;
            img = imread(strcat(IMG_DIR, num2str(i), '\', num2str(j), '\', img_name));  %读取每张图像
            img_gray = rgb2gray(img);
            if (IF_RESIZE)  %resize
                img_gray = floor(imresize(img_gray, [RESIZE_SIZE, RESIZE_SIZE], 'bilinear'));  %双线性插值（比最邻近插值效果好）              
            end
            img_cnt = img_cnt + 1;
            all_imgs_2d(img_cnt, :, :) = img_gray;
            all_labels(img_cnt, 1) = i - 1; %0-7
        end
    end
end

%打乱图像，并分配训练集和测试集
rand_num = randperm(ETH80_IMG, ETH80_IMG);  %返回一行从1到x的整数中的x个，且这x个数各不相同
for i = 1 : ETH80_IMG
    all_imgs_2d_rand(i, :, :) = all_imgs_2d(rand_num(i), :, :);
    all_labels_rand(i, 1) = all_labels(rand_num(i), 1);
end
for i = 1 : round(ETH80_IMG * TRAIN_RATIO)
    train_imgs_2d(i, :, :) = all_imgs_2d_rand(i, :, :);
    train_labels(i, 1) = all_labels_rand(i, 1);
end
for i = 1 : round(ETH80_IMG * (1 - TRAIN_RATIO))
    test_imgs_2d(i, :, :) = all_imgs_2d_rand(i + round(ETH80_IMG * TRAIN_RATIO), :, :);
    test_labels(i, 1) = all_labels_rand(i + round(ETH80_IMG * TRAIN_RATIO), 1);
end

%判断是否高斯滤波
%type= 'gaussian'，为高斯低通滤波器，模板有两个，sigma表示滤波器的标准差（单位为像素，默认值为 0.5），window表示模版尺寸，默认值为[3,3]
if (strcmp(PRE_PROC, 'Dog'))
    H1 = fspecial('gaussian', WINDOW, SIGMA1);
    H2 = fspecial('gaussian', WINDOW, SIGMA2);
    DiffGauss = H1 - H2;
    %滤波每张训练图像
    for n = 1 : round(ETH80_IMG * TRAIN_RATIO)
         dog_img = abs(imfilter(squeeze(train_imgs_2d(n, :, :)), DiffGauss, 'replicate'));   %对任意类型数组或多维图像进行滤波
         dog_img = mat2gray(dog_img);      % 将图像矩阵归一化0到1范围内(包括0和1)
         dog_img = floor(dog_img * 255);   % 再乘255变成正常灰度值    
         
         %检查Dog后的图像是否合理
%          subplot(1, 2, 1); imshow(squeeze(train_imgs_2d(n, :, :))/255); title('Ori');
%          subplot(1, 2, 2); imshow(dog_img/255); title('Dog');
%          debug_temp = 0;   %此处设置断点对比Dog前后图像     
         
         train_imgs_2d(n, :, :) = dog_img;     
    end
    %滤波每张测试图像
    for n = 1 : round(ETH80_IMG * (1 - TRAIN_RATIO))
         dog_img = abs(imfilter(squeeze(test_imgs_2d(n, :, :)), DiffGauss, 'replicate'));   %对任意类型数组或多维图像进行滤波
         dog_img = mat2gray(dog_img);      % 将图像矩阵归一化0到1范围内(包括0和1)
         dog_img = floor(dog_img * 255);   % 再乘255变成正常灰度值
         test_imgs_2d(n, :, :) = dog_img;
    end
end

%% generate csv 

folder_name = strcat(num2str(TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)), 'x', num2str(TriOp(IF_RESIZE, RESIZE_SIZE, ETH80_SIZE)));
TRAIN_ENCODING_SAVE_DIR = strcat('..\Gray\', folder_name, '\');


train_imgs_1d = zeros(round(ETH80_IMG * TRAIN_RATIO),RESIZE_SIZE*RESIZE_SIZE);
test_imgs_1d = zeros(round(ETH80_IMG * (1 - TRAIN_RATIO)),RESIZE_SIZE*RESIZE_SIZE);

for i = 1:round(ETH80_IMG * TRAIN_RATIO)
    train_imgs_1d(i,:) =  reshape(train_imgs_2d(i,:,:),1,[]);
end
for i = 1:ETH80_IMG * (1 - TRAIN_RATIO)
    test_imgs_1d(i,:) =  reshape(test_imgs_2d(i,:,:),1,[]);
end

eth80_train = [train_labels,train_imgs_1d];
eth80_test = [test_labels,test_imgs_1d];

writematrix(eth80_train,strcat(TRAIN_ENCODING_SAVE_DIR,'eth80_train.csv'));
writematrix(eth80_test,strcat(TRAIN_ENCODING_SAVE_DIR,'eth80_test.csv'));


%% visulization
img_idx = 10;
img = squeeze(train_imgs_2d(img_idx,:,:)); 

I = mat2gray(img);
I = kron(I,ones(32,32));
imshow(I);

