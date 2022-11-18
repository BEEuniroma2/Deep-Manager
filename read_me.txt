This is a guide to set up your envirnoment and use the Deep Manager Toolbox
The Toolbox implements the feature selection method described in ""

Requirements
- python >=3.8
- libraries inside requirements.txt
In order to properly install the required settings we suggest following these steps:
- download and install anaconda at https://www.anaconda.com/products/individual

- download and install git software at https://git-scm.com/downloads

After installed anaconda create a new environment.
- Open anaconda prompt
- to download the source code write down: git clone https://github.com/BEEuniroma2/Deep-Manager.git
in this way the entire github repository will be cloned to the specified path on your pc usually on the path "C:\Users\user_name\Deep-Manager"
the user_name folder name depends on the user name of the pc in which you installed the package

- write down: conda create -n "Deep_Manager_Environment" python=3.8.8
- if you already created this environment access it via: conda activate "Deep_Manager_Environment"
- Change directory to the path containing this read_me.txt file via: cd "C:\Users\user_name\Deep-Manager"
- write down: pip install -r requirements.txt


now you are ready to use the toolbox!
run the GUI file in the folder containing this read_me.txt as well as the GUI.py file
- write down: python GUI.py

Now you should see a window like the mainwindow.png in this folder

click on the combobox to select one of the three available modalities
- 2D brightfield
- 3D phase-contrast
- 3D fluorescence

Suppose you chose 2D brightfield.

Click the "Select the setting file" button
a window will be opened asking you to load a setting file.
You can find a typical setting file for each modality inside the folder Corresponding to
the chosen modality inside DATA for demonstration. The file is in txt format
Chosen the file the parameters can be modified clicking the "Change provided settings" button: another window will be
opened. You can change interactively all the parameters (If you want to use Transfer Learning features be sure to select an existing network and corresponding layer name). Thresholds used for Discriminative Power (DP) and Sensitivity (SENS) strongly depend on the application: hence you can tune this hyperparameters to obtain a non-empty and small set of features (usually from 10 to 100). Typical values for DP and SENS threshold are 0.6-0.7, and 0.1-0.2, respectively. Once you are satisfied you can apply the desired changes clicking the Update settings button.
Once you are satisfied you can apply the desired changes clicking the Update settings button.

Click the select path to data button. The selected directory must contain 2 folders each of which containing images
from one of the two classes. We provided an example for each modality inside the "DATA for demonstration" folder. 
In our example we must select the "IM-ACQ-1-2D_brightfield" folder.

After you selected modality, setting file and path to data a combobox will be enabled
allowing visualization of the alterations that will be performed with the provided parameters.

click on the combobox under the Label "Choose the feature kind" and select one of the two options
- Deep Features
- Traditional Features

Clicking the option "Deep Features" features from the chosen neural network and layer will be extracted.
Clicking the option "Traditional Features" classical features will be extracted from images. 
In the order The features are:
1. Mean pixel intensity
2. Median
3. Standard deviation
4. Quantile 10 %
5. Quantile 25 %
6. Quantile 75 %
7. Quantile 90 %
8. Max of the image
9. Min of the image
10. Entropy of the image
11-70 Feature vector containing Haralick texture descriptors 


If you want to save Normal and modified images you have to check the " Save normal and modified images"
checkbutton. If you do so to continue performing test you have to specify the folder and file name where to save
these datasets. Images will be saved in .mat format.

Now you are ready to perform tests, so click the "Perform Tests" button. All other buttons will be disabled and 
you will read a label remembering you which test is being performed on the right column.

After all tests are performed the "Visualize SENS vs DP scatter plot" will be enabled allowing you to 
visualize SENS-DP scatter plot. Every file you visualize will be saved in the folder where you are executing the program.

To go on you click the "Choose features" button.
Another window will be opened with a check button on every set of features. The final set of features will be the 
union of the checked sets of features. To perform this union you have to click the "Select features" button.
If you select an empty set you will receive a the following warning: 
'please select a non empty set of features. If you already selected all the features sets, consider changing the setting parameters'


After chosing features you will be asked to select a folder containing test images. Here the same considerations as for
the "Select path to data" button apply: the selected directory must contain 2 folders each of which containing images
from one of the two classes.

Then you will be asked to select a folder where to save all the .mat containing the results.

After these two last operations the "Save results" button will be enabled. 
Clicking it you will save all the results. 

The "Reset GUI" button allows you to restart the GUI

The BEE button on the bottom right links you to the Bioinspired Electronic Engineering group web page.
