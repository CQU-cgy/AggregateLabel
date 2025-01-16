% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @编写：wtx
% @更新日期：2022/1/1
% @概述：定义全局变量，保存至mat文件中供其它脚本使用
% @备注1：该文件用户谨慎修改
% @备注2：此方法每次添加新的变量需实时运行更新，这是它的一个不好的地方
% -----------------------------------------------------------------------------------------------------------------------------------------------------%
MAX_PIXEL_VALUE = 256;  %像素最大值（严格说是255，256是为了方便计算）

MNIST_TRAIN_IMG = 60000;  %MNIST和FASHION_MNIST拥有相同的数据集个数和图像尺寸
MNIST_TEST_IMG = 10000;
MNIST_SIZE = 28;
MNIST_NCLASS = 10;
DENOISING_THRESHOLD = 16;  %小于去噪阈值的像素值置零

ETH80_IMG = 3280;  %ETH80数据集图像总数
ETH80_SIZE = 256;  %ETH80数据集尺寸: 256x256三通道图像
ETH80_NCLASS = 8;  %ETH80数据集类别数
ETH80_NFOLDER_PER_CLASS = 10; %ETH80数据集每个类别下包含的文件夹数
ETH80_NIMG_PER_FOLDER = 41;   %ETH80数据集每个类别的每个文件夹下包含的图像数

CIFAR10_TRAIN_NBATCH = 5;        %CIFAR10原始训练集共包含5个batch
CIFAR10_TEST_NBATCH = 1;         %CIFAR10原始测试集共包含1个batch
CIFAR10_NIMG_PER_BATCH = 10000;  %CIFAR10每个batch包含图像数
CIFAR10_TRAIN_IMG = 50000;       %CIFAR10训练集图像总数
CIFAR10_TEST_IMG = 10000;        %CIFAR10测试集图像总数
CIFAR10_SIZE = 32;               %CIFAR10尺寸

DVS_RESIZE_SIZE = 16;            %该版本DVS数据集仅支持Resize成16x16

MNIST_DVS_NCELL = 10000;          %MNIST_DVS数据集cell总数
MNIST_DVS_TRAIN_MAX_SPIKE = 1e7;  %MNIST_DVS训练集最大脉冲数
MNIST_DVS_TEST_MAX_SPIKE = 1e7;   %MNIST_DVS测试集最大脉冲数

CARD_DVS_NCELL = 100;  %CARD_DVS数据集cell总数
CARD_SIZE = 32;        %CARD原始图像尺寸

POSTURE_DVS_NCELL = 484;
POSTURE_SIZE = 32;

N_MNIST_TRAIN_NCELL = 60000;
N_MNIST_TEST_NCELL = 10000;
N_MNIST_SIZE = 34;

XF_FACE_IMG = 380;               %先锋人脸数据集总数
XF_FACE_SIZE = 128;              %先锋人脸数据集尺寸：128x128三通道图像
XF_FACE_NCLASS = 19;             %先锋人脸数据集类别数
XF_FACE_NIMG_PER_FOLDER = 20;    %先锋人脸数据集每个类别文件夹下包含的图像数

GESTURE10_IMG = 3000;              %GESTURE10数据集图像总数
GESTURE10_SIZE = 32;              %GESTURE10数据集尺寸: 256x256三通道图像
GESTURE10_NCLASS = 10;            %GESTURE10数据集类别数
GESTURE10_NIMG_PER_FOLDER = 300;   %GESTURE10数据集每个类别的每个文件夹下包含的图像数

save(['.\Definition_pkg', '.mat'], 'MNIST_TRAIN_IMG', 'MNIST_TEST_IMG', 'DENOISING_THRESHOLD', 'MNIST_SIZE', 'MNIST_NCLASS', 'MAX_PIXEL_VALUE' ...
      , 'ETH80_IMG', 'ETH80_SIZE', 'ETH80_NCLASS', 'ETH80_NFOLDER_PER_CLASS', 'ETH80_NIMG_PER_FOLDER', 'CIFAR10_TRAIN_NBATCH' ...
      , 'CIFAR10_TEST_NBATCH', 'CIFAR10_NIMG_PER_BATCH', 'CIFAR10_TRAIN_IMG','CIFAR10_TEST_IMG', 'CIFAR10_SIZE', 'MNIST_DVS_NCELL' ...
      , 'MNIST_DVS_TRAIN_MAX_SPIKE', 'MNIST_DVS_TEST_MAX_SPIKE', 'DVS_RESIZE_SIZE', 'CARD_DVS_NCELL', 'CARD_SIZE', 'POSTURE_DVS_NCELL' ...
      , 'POSTURE_SIZE', 'N_MNIST_TRAIN_NCELL', 'N_MNIST_TEST_NCELL', 'N_MNIST_SIZE', 'XF_FACE_IMG', 'XF_FACE_SIZE', 'XF_FACE_NCLASS', 'XF_FACE_NIMG_PER_FOLDER' ...
      , 'GESTURE10_IMG', 'GESTURE10_SIZE', 'GESTURE10_NCLASS', 'GESTURE10_NIMG_PER_FOLDER');
  
  
  
  