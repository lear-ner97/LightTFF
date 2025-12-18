# LightTFF
This is the official implementation of our paper : "LightTFF: Lightweight Time-Frequency Dual Branch Framework for Mid-Term and Short-Term Energy Forecasting" <br>
## step-by-step guidelines to reproduce results
1-	Put all the files of this Github repository in the same folder <br>
2-	To ensure reproducibility of the results located in results folder, please run the code using the environment specified in environment.yml <br>
3-	Once the environment is set up, choose which forecasting scenario depending on three parameters: <br>
*dataset: specify the dataset’s name in lines 25 and 27 of the main file, and pay attention that the data path in line 26 is similar to your data path <br>
*input sequence length $L$: specify its value in line 128 of the main file (hyperparameters’ section), and choose a value from 96, 192, 336, 720. <br>
*future horizon length $H$: specify its value in line 38 of the main file, and choose a value from 96, 192, 336, 720. <br>
4-  Once your forecasting scenario is defined (dataset, L, H), go to its corresponding file in results folder. The results folder is composed of three folders, each one corresponding to a different input sequence length L. Inside each folder, you’ll find 8 files, corresponding to all possible combinations of datasets and future horizon lengths H. Choose a scenario and open its corresponding file. Then, do the following changes in main.py and LightTFF.py: <br>
•	In main.py: please fill out the values of the hyperparameters in lines 24 to 41 from the .txt file. If the .txt file says static =”conv”, then activate line38 (bias trend feature), and set the variable bias_trend to its corresponding value in the .txt file. Otherwise, if static==’ma’, comment line 38.<br>
•	In the file LightTFF.py, please follow the guidelines throughout the definition of the class Model. The class is clearly commented and distinguishes the two cases when static==’conv’ and static== ‘ma’. <br>
5- Run main.py. At the end, you should get the same results as in the .txt file. If not, it means that either you missed some step(s) in the previous process. <br>
*Expected output: test mse, test mae, train vs validation loss curves, prediction vs true test data plot <br>

## Explanation of the files (in alphabetic order): 
*acf.py: used to plot the Autocorrelation Function of each dataset <br>
*data_visualization.py: used to plot the evolution of electricity load and oil temperature during a full year <br>
*exp_basic.py: needed for cuda device setup <br>
*exp_main.py: contains the training, validation, and test functions <br>
*heatmap_weight_cptf.py: used to visualize the heatmap of the model’s Jacobian <br>
*LightTFF.py: implementation of our model’s class <br>
*main.py: used to train, validate, and test our model on different forecasting scenarios <br>
*spectrum_visualization.py: used to visualize the spectrum of the raw time series, the static component, and the dynamic component <br>
*The folders data_provider and utlis contain other useful functions needed for model training such as data loading and evaluations metrics. <br>
## Acknowledgement
We truly appreciate the importance of these github repositories  in the development of our framework:
* https://github.com/lss-1138/SparseTSF <br>
* https://github.com/showmeon/TimeEmb/tree/main/TimeEmb-main <br>
## 📬 Contact
If you have questions or encounter issues, please [open an issue](https://github.com/lear-ner97/LightTFF/issues) or contact us at **sami DOT benbrahim AT mail DOT concordia DOT ca**.

