AggregateLabel

This project involves the C# software code for an optimized AL algorithm (fixed-point version) as proposed in the article: An Edge Neuromorphic Processor With High-Accuracy On-Chip Aggregate-Label Learning, IEEE Transactions on Circuits and Systems II: Express Briefs, doi: 10.1109/TCSII.2025.3529670.  


Tools
Visual studio 2019 and above (preferably 2019)  
Matlab  
Python  


File Description
The root directory of the project is named AggregateLabel, inside which there are two folders.

1) DataSet&Proc_code: Image preprocessing script, including image scaling, DOG filtering and csv file generation. It has four sub-folders as:
① ETH80
② FASHION_MNIST
③ ORL
④ Yale

Inside each of the ETH, ORL, and YALE folders, there are three sub-folders:
① code: xxx2csv.m (eg. eth2csv) is the image preprocessing script, it will generate a csv file.
② Gray: Storing the CSV files.
③ Original_png：Store the original dataset's PNG files, since the file size limit of GitHub, we did not provide it. Readers can obtain it from other sources on the Internet.

In the FASHION_MNIST folder, ubyte2csv.py is a Python script used to obtain CSV files.

2) FixedPoint: algorithm implementation emulating hardware-implemented fixed-point data processing

we created five folders under the "FixedPoint" path for corresponding datasets used in the article:
① FixedtPoint_mnist
② FixedtPoint_fashion
③ FixedtPoint_ETH
④ FixedtPoint_ORL
⑤ FixedtPoint_YALE

Each of them is a complete VS project. Each of the VS project contains 4 main files. The related files are described as follows:

① Parameter.cs: Parameter file, which contains parameters that can be defined or configured.
② PoisionCoding.cs: Input spike encoding file, we use temporal coding in this project.
③ AL_MODEL.cs: The algorithm part, including training and testing, note that there are also some parameters that need to be re-configured in this file.
④ Program.cs: Main function entry.


How to run it
1) Download the project
①Download 5 volume zip files, which are respectively named AggregateLabel.zip, AggregateLabel.z01, AggregateLabel.z02, and so on.
② Select AggregateLabel.zip for unzipping, and you will get a folder named AggregateLabel, which is the root directory of the project.


2) Image preprocessing
When preprocessing the ETH, ORL, and YALE datasets, run the xxx2csv.m file located in the code folder (e.g., eth2csv), and the CSV files will be generated in the Gray folder.

Under FASHION_MNIST folder, run ubyte2csv.py in Python to get the csv file.


3) Model running
① Open any folder within FixedPoint folder, such as FixedtPoint_mnist.
② Click on file ‘AggregateLabel.sln’ in the FixedtPoint_mnist folder to open the VS project. 
③ Click to run and it will directly start the training process. 
During the training process, the terminal will print the accuracy of each epoch, as shown in the following example:
epoch:0 train acc: 92.0566666666667
epoch:0 test acc: 92.56
Train acc represents the accuracy on the training set, and test acc represents the accuracy on the test set. The highest accuracy is manually recorded. 


Pay attention
The algorithm code is readily executable. The negligible discrepancy (usually within -0.05% ~ +0.1%) between the accuracies obtained by this code and the ones reported in our article might be induced by the randomness in weight initialization and the training sample order perturbation. The mean and standard deviation of the accuracy are calculated by Excel.


