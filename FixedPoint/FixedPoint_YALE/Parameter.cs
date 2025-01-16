using System;
using System.Collections.Generic;
using System.Text;
using System.Collections;
using System.Linq;
using System.IO;

namespace AggregateLabel
{
    class Parameter
    {
        private const string version = "fixed_YALE";


        public const int NeuralCoreNum = 8;
        public static int[] NeuronNumPerNPT = { 64, 16 };
        public static int[] MaxEachLayerNeurons = { 784, 512, 128 };

        public static int[] MemIntBits = { 12, 12 }; // 整数位数 
        public static int[] MemFracBits = { 0, 8 }; // 小数位数
        public static int[] WeightIntBits = { 7, 7 }; // 整数位数
        public static int[] WeightFracBits = { 0, 8 }; // 小数位数
        public static int SpikeCountBits = 4; // 整数位数

        public static int LayersNum = 3;

        private const int ImageColumnSize = 28;      //图像宽 mnist:28 
        private const int ImageRowSize = 28;        //图像高 mnist:28 

        public static int InputLayerNeurons = ImageColumnSize * ImageRowSize;  //输入神经元的个数（按一维排列）
        public const int NumClass = 10;          //类别数 MNIST:10
        public const int EachClassNeuronNum = 10; //每类的神经元个数
        public const int OutputLayerNeurons = NumClass * EachClassNeuronNum;  //输出神经元
        public const int HiddenLayerNeurons = 512;
        public static int[] EachLayerNeurons = { InputLayerNeurons, HiddenLayerNeurons, OutputLayerNeurons };

        public static int[][,] WeightTensor = new int[LayersNum - 1][,];  //权重矩阵

        public static int[] NeuronLabel = new int[Parameter.OutputLayerNeurons];
        public static int ClassTaregtSpikeOutNum = 3; //目标输出脉冲数
        public static int SilentSpikeOutNum = 0; //目标输出脉冲数

        public static int lr_rsb = 15;
        public static int lr_hidden_rsb = 16;

        public const int SampleTimeWindow = 64;
        public static int[] InitTreshold = { 6 << (MemFracBits[0] + 7), 6 << (MemFracBits[1] + 7) };

        public static bool[][,] RandErrMatrix = new bool[LayersNum - 2][,]; // DFA反馈矩阵

        public static int[] MemUpperBound = { (1 << (MemIntBits[0] + MemFracBits[0])) - 1, (1 << (MemIntBits[1] + MemFracBits[1])) - 1 };
        public static int[] MemLowBound = { -(1 << (MemIntBits[0] + MemFracBits[0])), -(1 << (MemIntBits[1] + MemFracBits[1])) };

        public static int[] WeightUpperBound = { 1 * (1 << 7) - 1, 1 * (1 << 15) - 1 };
        public static int[] WeightLowBound = { -1 * (1 << 7), -1 * (1 << 15) };

        public static int SpikeCountBound = (1 << SpikeCountBits) - 1;

        public static int UpdateStartLayer = 1; //2是输出层

        public static int[] reg_data = new int[NeuralCoreNum];
        public static int RandSeed = 11;

        // 统计用
        public static int CorrectPredict = 0; //总计正确的个数
    }
}
