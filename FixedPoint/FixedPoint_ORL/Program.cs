using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using CODING;

namespace AggregateLabel
{
    class Program
    {

        public static string[] CodingTheme = new string[] { "PoisionCoding", "MultiSpikeCoing", "TemporalCoding" };
        public static string ResultPath = "../../Result/AL.txt";

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            /*************** MNIST编码 **********************/
            PoisionCoding.Coding(CodingTheme[2]);
            Console.WriteLine("Coding finished!");
            /************脉冲数据的读取和权重的初始化*********/
            AL_MODEL.PriorProcess();
            //AL_MODEL.WeightReadInit();
            AL_MODEL.WriteSampleLabel();
            /*************** 训练 **********************/

            AL_MODEL.Train();

            //AL_MODEL.Test(0);

            /*************** END *****************/
            Console.WriteLine("END");
            while (true)
            { Console.Read(); }


        }


    }
}
