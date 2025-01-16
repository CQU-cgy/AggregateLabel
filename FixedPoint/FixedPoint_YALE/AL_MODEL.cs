using System;
using System.Collections.Generic;
using System.Text;
using System.Collections;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics;
using CODING;

namespace AggregateLabel
{
    class AL_MODEL
    {

        public static int MAX_EPOCH = 200;
        public const int MNISTTrainNum = PoisionCoding.WriterFileTrainNum; //文件所保存的长度
        public const int MNISTTestNum = PoisionCoding.WriterFileTestNum;
        public static int TrainNum = 383; //实际用于训练的样本量
        public static int TestNum = 255;


        public const string InitWeightPath = PoisionCoding.DataPath + "init_weight.txt";

        public static int[][] TrainCodingSpike = new int[MNISTTrainNum][]; //Aers
        public static int[][] TestCodingSpike = new int[MNISTTestNum][];
        public static int[] TrainLabel = new int[MNISTTrainNum];
        public static int[] TestLabel = new int[MNISTTestNum];
        public static bool[][,] TrainSpikeMap = new bool[MNISTTrainNum][,]; //SpikeMap
        public static bool[][,] TestSpikeMap = new bool[MNISTTestNum][,];
        public static Random WeightRandomHandle = new Random(31); //设置随机数种子
        public static Random UpdateRandomHandle = new Random(1);  //用于随机更新的随机种子


        //文件读取句柄（ReaderStream）
        public const string TrainCodingSpikePath = PoisionCoding.DataPath + "mnist_train_coding.txt";
        public const string TrainLabelPath = PoisionCoding.DataPath + "mnist_train_label.txt";
        public const string TestCodingSpikePath = PoisionCoding.DataPath + "mnist_test_coding.txt";
        public const string TestLabelPath = PoisionCoding.DataPath + "mnist_test_label.txt";
        static FileStream fs_train_data = new FileStream(TrainCodingSpikePath, FileMode.Open);    //打开训练集文件
        static StreamReader sr_train_data = new StreamReader(fs_train_data);
        static FileStream fs_train_label = new FileStream(TrainLabelPath, FileMode.Open);
        static StreamReader sr_train_label = new StreamReader(fs_train_label);
        static FileStream fs_test_data = new FileStream(TestCodingSpikePath, FileMode.Open);    //打开测试集文件
        static StreamReader sr_test_data = new StreamReader(fs_test_data);
        static FileStream fs_test_label = new FileStream(TestLabelPath, FileMode.Open);
        static StreamReader sr_test_label = new StreamReader(fs_test_label);



        public static void GlobalClear()
        {
            Parameter.CorrectPredict = 0; // 统计清零
        }

        // 数据读取及权重初始化
        public static void PriorProcess()
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            for (int npt_idx = 0; npt_idx < Parameter.NeuralCoreNum; npt_idx++)
            {
                Many2OneLFSR(Parameter.RandSeed, 0, ref Parameter.reg_data[npt_idx], 8);
            }
            WeightRandomInit();
            ReadEvenSpikeFromFile();
            NeuronLabelInit();
            RandErrMatrixInit();


            watch.Stop();
            Console.WriteLine("读取文件结束，读取文件共消耗 {0} 分钟", (watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
            watch.Reset();
        }

        // DFA反馈矩阵初始化
        public static void RandErrMatrixInit()
        {
            for (int LayerFlag = 1; LayerFlag < Parameter.LayersNum - 1; LayerFlag++)
            {
                Parameter.RandErrMatrix[LayerFlag - 1] = new bool[Parameter.OutputLayerNeurons, Parameter.EachLayerNeurons[LayerFlag]];
                for (int o_addr = 0; o_addr < Parameter.OutputLayerNeurons; o_addr++)
                {
                    for (int h_addr = 0; h_addr < Parameter.EachLayerNeurons[LayerFlag]; h_addr++)
                    {
                        if (WeightRandomHandle.NextDouble() > 0.5)
                            Parameter.RandErrMatrix[LayerFlag - 1][o_addr, h_addr] = true;
                        else
                            Parameter.RandErrMatrix[LayerFlag - 1][o_addr, h_addr] = false;
                    }
                }
            }

            string RndMatrixPath = PoisionCoding.DataPath + "rand_matrix.txt";
            using (StreamWriter RndMatrixRecord = new StreamWriter(RndMatrixPath))
            {
                for (int LayerFlag = 1; LayerFlag < Parameter.LayersNum - 1; LayerFlag++)
                {
                    for (int o_addr = 0; o_addr < Parameter.OutputLayerNeurons; o_addr++)
                    {
                        for (int i_addr = 0; i_addr < Parameter.EachLayerNeurons[LayerFlag]; i_addr++)
                        {
                            RndMatrixRecord.WriteLine(Convert.ToString(Parameter.RandErrMatrix[LayerFlag - 1][o_addr, i_addr]));
                        }
                    }
                }
            }

            string RndMatrixHexPath = PoisionCoding.DataPath + "rand_matrix_hex.txt";
            using (StreamWriter RndMatrixRecord = new StreamWriter(RndMatrixHexPath))
            {
                for (int LayerFlag = 1; LayerFlag < Parameter.LayersNum - 1; LayerFlag++)
                {
                    for (int i_addr = 0; i_addr < Parameter.EachLayerNeurons[LayerFlag]; i_addr++)//注意保存顺序，和硬件相对应
                    {
                        for (int o_addr = 0; o_addr < Parameter.OutputLayerNeurons; o_addr++)
                        {
                            if (Parameter.RandErrMatrix[LayerFlag - 1][o_addr, i_addr] == true)
                                RndMatrixRecord.WriteLine(Convert.ToString(1, 16));
                            if (Parameter.RandErrMatrix[LayerFlag - 1][o_addr, i_addr] == false)
                                RndMatrixRecord.WriteLine(Convert.ToString(0, 16));

                        }
                    }
                }
            }
        }

        // 神经元标签初始化
        public static void NeuronLabelInit()
        {
            for (int o_addr = 0; o_addr < Parameter.OutputLayerNeurons; o_addr++)
            {
                Parameter.NeuronLabel[o_addr] = (int)(o_addr / Parameter.EachClassNeuronNum); //设为固定值
            }
            string NeuronLabelPath = PoisionCoding.DataPath + "neuron_label.txt";
            using (StreamWriter NeuronLabelRecord = new StreamWriter(NeuronLabelPath))
            {
                for (int o_addr = 0; o_addr < Parameter.OutputLayerNeurons; o_addr++)
                {
                    NeuronLabelRecord.WriteLine(Convert.ToString(Parameter.NeuronLabel[o_addr]),16);
                }

            }
        }

        // 记录神经元标签
        public static void WriteSampleLabel()
        {
            string SampleLabelPath = PoisionCoding.DataPath + "sample_label.txt";
            using (StreamWriter SampleLabelRecord = new StreamWriter(SampleLabelPath))
            {
                for (int i = 0; i < TrainNum; i++)
                {
                    SampleLabelRecord.WriteLine(Convert.ToString(TrainLabel[i]), 16);
                }

            }
        }


        // 权重初始化
        public static void WeightRandomInit()
        {
            using (StreamWriter InitWeightRecord = new StreamWriter(InitWeightPath))
            {
                for (int layer = 0; layer < Parameter.LayersNum - 1; layer++)
                {
                    Parameter.WeightTensor[layer] = new int[Parameter.EachLayerNeurons[layer + 1], Parameter.EachLayerNeurons[layer]];
                    for (int i = 0; i < Parameter.WeightTensor[layer].GetLength(0); i++)
                    {
                        for (int j = 0; j < Parameter.WeightTensor[layer].GetLength(1); j++)
                        {
                            Parameter.WeightTensor[layer][i, j] = (int)Math.Round((WeightRandomHandle.NextDouble() * 2 - 1) * Math.Sqrt(6) / Math.Sqrt(Parameter.EachLayerNeurons[layer + 1] + Parameter.EachLayerNeurons[layer]) * (1 << (Parameter.WeightFracBits[layer] + 7)));
                            InitWeightRecord.WriteLine(Convert.ToString(Parameter.WeightTensor[layer][i, j]));
                        }
                    }
                }
            }
        }

        // 读取储存的AER文件
        public static void ReadEvenSpikeFromFile()
        {
            string line_data;
            int count = 0;
            while ((line_data = sr_train_label.ReadLine()) != null)  //训练标签
            {
                TrainLabel[count++] = Convert.ToInt32(line_data);
            }
            count = 0;
            while ((line_data = sr_test_label.ReadLine()) != null)  //测试标签
            {
                TestLabel[count++] = Convert.ToInt32(line_data);
            }
            count = 0;
            while ((line_data = sr_train_data.ReadLine()) != null)
            {
                string[] str_data = line_data.Split(' ');

                TrainCodingSpike[count] = new int[str_data.Length - 1];

                for (int i = 0; i < str_data.Length - 1; i++)
                {
                    TrainCodingSpike[count][i] = Convert.ToInt32(str_data[i]);
                }
                count++;
            }
            count = 0;
            while ((line_data = sr_test_data.ReadLine()) != null)
            {
                string[] str_data = line_data.Split(' ');
                TestCodingSpike[count] = new int[str_data.Length - 1];

                for (int i = 0; i < str_data.Length - 1; i++)
                {
                    TestCodingSpike[count][i] = Convert.ToInt32(str_data[i]);
                }
                count++;
            }
        }

        // 读取权重及DFA反馈矩阵
        public static void WeightReadInit()
        {
            string read_weight_path = PoisionCoding.DataPath + "weight0.txt";
            using (StreamReader ReadWeight = new StreamReader(read_weight_path))
            {
                for (int layer = 0; layer < Parameter.LayersNum - 1; layer++)
                {
                    for (int i = 0; i < Parameter.WeightTensor[layer].GetLength(0); i++)
                    {
                        for (int j = 0; j < Parameter.WeightTensor[layer].GetLength(1); j++)
                        {
                            Parameter.WeightTensor[layer][i, j] = Convert.ToInt32(ReadWeight.ReadLine());
                        }
                    }
                }
            }

            string read_rndt_path = PoisionCoding.DataPath + "rand_matrix.txt";
            using (StreamReader ReadWeight = new StreamReader(read_rndt_path))
            {
                for (int layer = 0; layer < Parameter.LayersNum - 2; layer++)
                {
                    for (int i = 0; i < Parameter.RandErrMatrix[layer].GetLength(0); i++)
                    {
                        for (int j = 0; j < Parameter.RandErrMatrix[layer].GetLength(1); j++)
                        {
                            Parameter.RandErrMatrix[layer][i, j] = Convert.ToBoolean(ReadWeight.ReadLine());
                        }
                    }
                }
            }
        }

        // 记录权重
        public static void WriteWeight(int epoch)
        {
            string WEIGHT_PATH = PoisionCoding.DataPath + "weight" + epoch.ToString() + ".txt";
            using (StreamWriter PureWeightRecord = new StreamWriter(WEIGHT_PATH))
            {
                for (int layer = 0; layer < Parameter.LayersNum - 1; layer++) 
                {
                    for (int i = 0; i < Parameter.WeightTensor[layer].GetLength(0); i++)
                    {
                        for (int j = 0; j < Parameter.WeightTensor[layer].GetLength(1); j++)
                        {
                            PureWeightRecord.WriteLine(Parameter.WeightTensor[layer][i, j]);
                        }
                    }
                }
            }

            string HEX_WEIGHT_PATH = PoisionCoding.DataPath + "hex_weight" + epoch.ToString() + ".txt";
            using (StreamWriter PureWeightRecord = new StreamWriter(HEX_WEIGHT_PATH))
            {
                for (int layer = 0; layer < Parameter.LayersNum - 1; layer++)
                {
                    for (int i = 0; i < Parameter.WeightTensor[layer].GetLength(0); i++)
                    {
                        for (int j = 0; j < Parameter.WeightTensor[layer].GetLength(1); j++)
                        {
                            PureWeightRecord.WriteLine(Convert.ToString(Parameter.WeightTensor[layer][i, j], 16));
                        }
                    }
                }
            }
        }

        // 训练过程
        public static void Train()
        {
            Stopwatch watch = new Stopwatch(); //计时

            for (int epoch = 0; epoch < MAX_EPOCH; epoch++)
            {
                watch.Start();

                GlobalClear();
                for (int i = 0; i < TrainNum; i++)
                {
                    ModelProcess(TrainCodingSpike[i], TrainLabel[i], true, i);
                    if ((i + 1) % TrainNum == 0)
                    {
                        Console.WriteLine("s {0} s", (i + 1));
                        Test(epoch);
                    }
                }

                watch.Stop();
                Console.WriteLine("共消耗 {0} 分钟", (watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
                watch.Reset();

                WriteWeight(epoch);
            }
        }

        // 测试过程
        public static void Test(int epoch)
        {
            Stopwatch watch = new Stopwatch(); //计时
            watch.Start();
            /* 测试 */
            GlobalClear();
            for (int i = 0; i < TrainNum; i++)
            {
                ModelProcess(TrainCodingSpike[i], TrainLabel[i], false, i);
            }
            double train_acc = (double)(Parameter.CorrectPredict) / TrainNum * 100;
            Console.WriteLine("epoch:{0} train acc: {1}", epoch, train_acc);

            GlobalClear();
            for (int i = 0; i < TestNum; i++)
            {
                ModelProcess(TestCodingSpike[i], TestLabel[i], false, i);
            }
            double test_acc = (double)(Parameter.CorrectPredict) / TestNum * 100;
            Console.WriteLine("epoch:{0} test acc: {1}", epoch, test_acc);
            watch.Stop();
            Console.WriteLine("测试消耗 {0} 分钟", (watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
            watch.Reset();
        }


        public static void SingleLayer(int Layer, List<int> InputList, ref List<int> OutList, ref bool[][,] SpikeMap, int TimeStep, ref int[][] NeuronMemPotential, ref int[][] Vpeak, ref int[][] Tpeak,
                                        ref int[][] TotalSpikeOutNum, int InNodes, int OutNodes, bool mode)
        {
            int AersPtr = 0;
            while (AersPtr < InputList.Count)
            {
                int i_addr = InputList[AersPtr];

                SpikeMap[Layer - 1][TimeStep, i_addr] = true;

                for (int o_addr = 0; o_addr < OutNodes; o_addr++)
                {
                    NeuronMemPotential[Layer - 1][o_addr] += Parameter.WeightTensor[Layer - 1][o_addr, i_addr]; // 小数位数相同，直接相加
                    NeuronMemPotential[Layer - 1][o_addr] = Math.Min(Parameter.MemUpperBound[Layer - 1], Math.Max(NeuronMemPotential[Layer - 1][o_addr], Parameter.MemLowBound[Layer - 1]));
                }
                AersPtr++;
            }

            for (int o_addr = 0; o_addr < OutNodes; o_addr++)
            {
                if (Vpeak[Layer - 1][o_addr] < NeuronMemPotential[Layer - 1][o_addr])
                {
                    Tpeak[Layer - 1][o_addr] = TimeStep;
                    Vpeak[Layer - 1][o_addr] = NeuronMemPotential[Layer - 1][o_addr];
                }
                if (NeuronMemPotential[Layer - 1][o_addr] > (TotalSpikeOutNum[Layer - 1][o_addr] + 1) * Parameter.InitTreshold[Layer - 1])
                {
                    TotalSpikeOutNum[Layer - 1][o_addr] += 1;
                    TotalSpikeOutNum[Layer - 1][o_addr] = Math.Min(TotalSpikeOutNum[Layer - 1][o_addr], Parameter.SpikeCountBound);
                    OutList.Add(o_addr);
                }
            }
        }

        public static void ModelProcess(int[] SpikeList, int InputClassLabel, bool mode, int ImgNum) // MODE:ture训练
        {
            int TimeStep = 0;
            int AersPtr = 0;

            /* 变量初始化 */
            bool[][,] SpikeMap = new bool[Parameter.LayersNum][,];
            for (int layer = 0; layer < Parameter.LayersNum; layer++)
                SpikeMap[layer] = new bool[Parameter.SampleTimeWindow,Parameter.EachLayerNeurons[layer]];
            int[][] NeuronMemPotential = new int[Parameter.LayersNum - 1][]; //膜电位            
            int[][] TotalSpikeOutNum = new int[Parameter.LayersNum - 1][]; //实际输出脉冲数
            int[][] Vpeak = new int[Parameter.LayersNum - 1][];
            int[][] Tpeak = new int[Parameter.LayersNum - 1][];
            for (int layer = 1; layer < Parameter.LayersNum; layer++)
            {
                NeuronMemPotential[layer - 1] = new int[Parameter.EachLayerNeurons[layer]];
                TotalSpikeOutNum[layer - 1] = new int[Parameter.EachLayerNeurons[layer]];
                Vpeak[layer - 1] = new int[Parameter.EachLayerNeurons[layer]];

                Tpeak[layer - 1] = new int[Parameter.EachLayerNeurons[layer]];
                for (int addr = 0; addr < Parameter.EachLayerNeurons[layer]; addr++)
                { Tpeak[layer - 1][addr] = Parameter.SampleTimeWindow; }
            }


            while (TimeStep < Parameter.SampleTimeWindow)
            {
                List<int> InList = new List<int>();
                List<int> OutList = new List<int>();
                while (AersPtr < SpikeList.Length && SpikeList[AersPtr] <= TimeStep)
                {
                    int i_addr = SpikeList[AersPtr + 1];
                    AersPtr += 2;
                    OutList.Add(i_addr);
                }

                /***** Each Layer*****/
                for (int layer = 1; layer < Parameter.LayersNum; layer++)
                {
                    InList = OutList;
                    OutList = new List<int>();
                    SingleLayer(layer, InList, ref OutList, ref SpikeMap, TimeStep, ref NeuronMemPotential, ref Vpeak, ref Tpeak,
                                        ref TotalSpikeOutNum, Parameter.EachLayerNeurons[layer - 1], Parameter.EachLayerNeurons[layer], mode);
                }

                /***** time-step increase*****/
                TimeStep++;
            }

            if (mode == true)
            {
                Update(InputClassLabel, Tpeak, Vpeak, TotalSpikeOutNum,SpikeMap);
            }
            else
            {
                WinnerDecide(TotalSpikeOutNum, InputClassLabel, NeuronMemPotential);
            }
        }


        // 权重更新
        public static void Update(int InputClassLabel, int[][] Tpeak, int[][] Vpeak, int[][] TotalSpikeOutNum, bool[][,] SpikeMap)
        {
            int[][] LayerErr = new int[Parameter.LayersNum - 1][];
            for (int layer = 0; layer < Parameter.LayersNum - 1; layer++) //初始化各层误差
            { 
                LayerErr[layer] = new int[Parameter.EachLayerNeurons[layer + 1]]; 
            }

            /* Layer 2 err */
            int LayerFlag_2 = Parameter.LayersNum - 2; //最后一层的误差;
            for (int o_addr = 0; o_addr < Parameter.OutputLayerNeurons; o_addr++)
            {
                int TargetSpikeNum = (InputClassLabel == Parameter.NeuronLabel[o_addr]) ? Parameter.ClassTaregtSpikeOutNum : 0;
                /* 误差计算 */
                if (TotalSpikeOutNum[LayerFlag_2][o_addr] < TargetSpikeNum)
                {
                    LayerErr[LayerFlag_2][o_addr] = Vpeak[LayerFlag_2][o_addr] - Parameter.InitTreshold[LayerFlag_2] * TargetSpikeNum;
                }
                else if (TotalSpikeOutNum[LayerFlag_2][o_addr] > TargetSpikeNum)
                {
                    LayerErr[LayerFlag_2][o_addr] = Vpeak[LayerFlag_2][o_addr] - Parameter.InitTreshold[LayerFlag_2] * (TargetSpikeNum + 1);
                }
            }

            /* Layer 1 err */
            for (int o_addr = 0; o_addr < Parameter.OutputLayerNeurons; o_addr++)
            {
                for (int h_addr = 0; h_addr < Parameter.EachLayerNeurons[1]; h_addr++)
                {
                    if (Parameter.RandErrMatrix[0][o_addr, h_addr]) //误差传播
                        LayerErr[0][h_addr] += LayerErr[LayerFlag_2][o_addr] >> 7;
                    else
                        LayerErr[0][h_addr] -= LayerErr[LayerFlag_2][o_addr] >> 7;
                }
            }

            /* Layer 2 weight update */
            int LayerFlag = 2;
            for (int o_addr = 0; o_addr < Parameter.OutputLayerNeurons; o_addr++)
            {
                for (int h_addr = 0; h_addr < Parameter.EachLayerNeurons[1]; h_addr++)
                {
                    //pop count 在硬件上是并行计算的
                    int Q_tpeak = 0;
                    if (Tpeak[LayerFlag - 1][o_addr] != Parameter.SampleTimeWindow)
                        for (int time = 0; time <= Tpeak[LayerFlag - 1][o_addr]; time++)
                            if (SpikeMap[LayerFlag - 1][time, h_addr] == true)
                                Q_tpeak += 1;

                    int temp = checked(LayerErr[LayerFlag - 1][o_addr] * Q_tpeak);
                    Parameter.WeightTensor[LayerFlag - 1][o_addr, h_addr] -= (int)(temp >> Parameter.lr_rsb); // 直接截断
                    Parameter.WeightTensor[LayerFlag - 1][o_addr, h_addr] = Math.Min(Parameter.WeightUpperBound[LayerFlag - 1], Math.Max(Parameter.WeightTensor[LayerFlag - 1][o_addr, h_addr], Parameter.WeightLowBound[LayerFlag - 1]));
                }
            }

            /* Layer 1 weight update */
            LayerFlag = 1;
            for (int h_addr = 0; h_addr < Parameter.EachLayerNeurons[LayerFlag]; h_addr++)
            {
                for (int i_addr = 0; i_addr < Parameter.EachLayerNeurons[LayerFlag - 1]; i_addr++)
                {
                    int Q_tpeak = 0;
                    if (Tpeak[LayerFlag - 1][h_addr] != Parameter.SampleTimeWindow)
                        for (int time = 0; time <= Tpeak[LayerFlag - 1][h_addr]; time++)
                            if (SpikeMap[LayerFlag - 1][time, i_addr] == true)
                                Q_tpeak += 1;
                    long temp = checked(LayerErr[LayerFlag - 1][h_addr] * Q_tpeak);

                    int NPTidx = h_addr / Parameter.NeuronNumPerNPT[0];
                    Many2OneLFSR(Parameter.RandSeed, 1, ref Parameter.reg_data[NPTidx], 8);
                    int random_data = Parameter.reg_data[NPTidx] << (Parameter.lr_hidden_rsb - 8); //17-8
                    long WeightDeltaInPrecision = temp >> (Parameter.lr_hidden_rsb);
                    long WeightDeltaOutPrecision = temp - (WeightDeltaInPrecision << Parameter.lr_hidden_rsb);
                    Parameter.WeightTensor[LayerFlag - 1][h_addr, i_addr] -= (int)((WeightDeltaInPrecision + ((WeightDeltaOutPrecision > random_data) ? 1 : 0)));

                    Parameter.WeightTensor[LayerFlag - 1][h_addr, i_addr] = Math.Min(Parameter.WeightUpperBound[LayerFlag - 1], Math.Max(Parameter.WeightTensor[LayerFlag - 1][h_addr, i_addr], Parameter.WeightLowBound[LayerFlag - 1]));
                }
            }
        }

        // 预测结果判决
        public static void WinnerDecide(int[][] TotalSpikeOutNum, int InputClassLabel, int[][] NeuronMemPotential)
        {
            Random rnd_choice = new Random(0);

            int predict = -1;
            double[] EachClassSpikeNum = new double[Parameter.NumClass];
            double[] EachClassMemSum = new double[Parameter.NumClass];

            for (int i = 0; i < Parameter.NumClass; i++)
            {
                for (int j = 0; j < Parameter.EachClassNeuronNum; j++)
                {
                    int o_addr = i * Parameter.EachClassNeuronNum + j;
                    EachClassSpikeNum[i] += TotalSpikeOutNum[Parameter.LayersNum - 2][o_addr];
                    EachClassMemSum[i] += NeuronMemPotential[Parameter.LayersNum - 2][o_addr];
                }
            }
            for (int i = 0; i < Parameter.NumClass; i++)
            {
                EachClassSpikeNum[i] = EachClassSpikeNum[i] / Parameter.EachClassNeuronNum / Parameter.ClassTaregtSpikeOutNum; //归一化
                EachClassSpikeNum[i] -= 1; // 调整到0为中心
                EachClassSpikeNum[i] = Math.Abs(EachClassSpikeNum[i]);

                EachClassMemSum[i] -= (double)Parameter.InitTreshold[Parameter.LayersNum - 2] * Parameter.ClassTaregtSpikeOutNum * Parameter.EachClassNeuronNum;
                EachClassMemSum[i] = Math.Abs(EachClassMemSum[i]);
            }

            int equal_num = 0;
            List<int> equal_idx = new List<int>();
            for (int i = 0; i < Parameter.NumClass; i++)
            {
                if (EachClassSpikeNum[i] == EachClassSpikeNum.Min())
                {
                    equal_num += 1;
                    equal_idx.Add(i);
                }
            }
            if (equal_num == 1)
            {
                predict = equal_idx[0];
                if (predict == InputClassLabel)
                    Parameter.CorrectPredict += 1;
            }
            else
            {
                int idx = rnd_choice.Next(0, equal_idx.Count); //随机猜一个
                predict = equal_idx[idx];
                if (predict == InputClassLabel)
                    Parameter.CorrectPredict += 1;
            }
        }

        /* ******************************根据LFSR反馈多项式列表查询************************************
         * 触发器的个数        反馈多项式                           随机数序列周期        
         *     2               x^2 + x +1                                 3                  
         *     3               x^3 + x^2 + 1                              7
         *     4               x^4 + x^3 + 1                              15
         *     5               x^5 + x^3 + 1                              31
         *     6               x^6 + x^5 + 1                              63
         *     7               x^7 + x^6 + 1                             127
         *     8               x^8 + x^6 +x^5 + x^4 + 1                  255
         *     9               x^9 + x^5 + 1                             511
         *     10              x^10 + x^7 + 1                            1023
         *     11              x^11 + x^9 + 1                            2047
         *     12              x^12 + x^11 +x^10 + x^4 + 1               4095
         *     13              x^13 + x^12 +x^11 + x^8 + 1               8191
         *     14              x^14 + x^13 +x^12 + x^2 + 1               16383
         *     15              x^15 + x^14 + 1                           32767
         *     16              x^16 + x^14 +x^13 + x^11 + 1              65535
         *     21              x^21 + x^19 + 1                          
         **********************************************************************************************     
        */
        //相比于传统的斐波那契LFSR，此处改进：一个脉冲驱动一次，而不是在每个时钟到来都要驱动
        //支持多bit输入，根据多bit数进行多项式更新
        public static void Many2OneLFSR(int seed_data, int init_state, ref int reg_data, int REG_BIT_NUM) //reg_data用来暂存上一次更新的值
        {
            if (init_state == 0) //初始状态，模拟硬件中的复位
            {
                reg_data = seed_data;
            }
            else
            {
                int LSB = 0;
                switch (REG_BIT_NUM)
                {
                    case 3: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 2)) & 1); break;
                    case 4: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 2)) & 1); break;
                    case 5: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 3)) & 1); break;
                    case 6: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 2)) & 1); break;
                    case 7: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 2)) & 1); break;
                    case 8: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 3)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 4)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 5)) & 1); break;
                    case 9: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 5)) & 1); break;
                    case 10: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 4)) & 1); break;
                    case 11: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 3)) & 1); break;
                    case 12: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 2)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 3)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 9)) & 1); break;
                    case 13: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 2)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 3)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 6)) & 1); break;
                    case 14: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 2)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 3)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 13)) & 1); break;
                    case 15: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 2)) & 1); break;
                    case 16: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 3)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 4)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 6)) & 1); break;
                    case 22: LSB = ((reg_data >> (REG_BIT_NUM - 1)) & 1) ^ ((reg_data >> (REG_BIT_NUM - 3)) & 1); break;
                    default: break;
                }
                reg_data = ((reg_data & ((1 << (REG_BIT_NUM - 1)) - 1)) << 1) + LSB;
            }
        }


    }
}




