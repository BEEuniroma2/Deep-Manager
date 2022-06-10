# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:46:12 2021

@author: Michele D'Orazio
———————————————————————————
Michele D'Orazio, PhD Student
Dept. Electronic Engineering

Interdisciplinary Center of Advanced Study of 
Organ-on-Chip and Lab-on-Chip Applications (IC-LOC)

University of Rome Tor Vergata
Via del politecnico 1, 00133 Roma, Italy
tel: +39 3409986753

"""

# import necessary libraries
from utils import select_network, construct_transfer_learning_model, read_all, my_read_resize
from utils import my_feature_selection, test_luminance_lamp, test_out_of_focus, test_movement,test_autofluo_luminance_lamp, test_gel_pattern, test_photobleaching, test_saturation
from utils import extract_standard_descriptors
import tensorflow as tf
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Combobox
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from imutils import paths
import webbrowser
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
icon_name = 'BEE.ico'


plt.ioff()
padx_0 = 60
pady_0 = 5
padx_1 = 60
pady_1 = 5
row_trasl = 0
class GUI:
    def __init__(self):
        # initialize window variable
        self.window = None
        # initialize as empty strings variables
        self.var_name_modality = ''
        self.var_setting_file = ''
        self.var_class_path = ''
        self.var_normal_dataset_path = ''
        self.var_modified_dataset_path = ''
        self.normal_images_path = ''
        self.modified_images_path = ''
        self.deep_features = ''
        self.traditional_features = ''
        # specify the web site for the button
        self.url = "https://web.bee.uniroma2.it/team-members/"
        
        # open new webbrowser window
        self.new = 1
        # initialize setting variables as None
        self.var_lev_th_fluo_ = None
        self.var_lev_perc_bleach_ = None
        self.var_lev_lambda_pattern_ = None
        self.var_lev_focus_ = None
        self.var_lev_trasl_ = None
        self.var_lev_rot_ = None
        self.var_lev_bright_ = None
        self.var_layer_name_ = None
        self.var_net_name_ = None
        
    def set_gui(self):
        # initialize a new window
        self.window = Tk()
        
        # set the window title
        self.window.title('Deep Manager')
        # set the window geometry widthxheight+x+y where x is the displacement on the orizontal axis
        # and y is the displacement on the veretical axis with respect to the top left corner 
        self.window.geometry("800x700+10+20")    
        # self.window.resizable(width=False, height=False)
        #Configure Rows and column
        Grid.columnconfigure(self.window,0,weight=1)
        Grid.columnconfigure(self.window,1,weight=1)
        padx_0 = 60
        pady_0 = 5
        padx_1 = 60
        pady_1 = 5
        row_trasl = 0
        for r in range(19+row_trasl):
            Grid.rowconfigure(self.window,r,weight=1, minsize=20)
            
        # put a label for the modaility combobox
        self.labelTop = Label(self.window,text="Choose the modality")
        self.labelTop.grid(column=0,row=1+row_trasl,padx=padx_0,pady=pady_0,sticky="nsew")
        # instantiate a tuple for the string of the three modalities
        data=("2D brightfield", "3D phase-contrast", "3D fluorescence")
        # create the combobox for the modality selection
        self.cb=Combobox(self.window, values=data)
        self.cb.grid(column=0, row=2+row_trasl,sticky="nsew",padx=padx_0,pady=pady_0)
        # bind the selection to a callback function
        self.cb.bind("<<ComboboxSelected>>",self.callbackFunc)
        # create a Button for changing the settings provided with the "Select setting file" button
        self.btn1 = Button(self.window, text='Change provided settings',command=self.make_new_window,state=DISABLED)
        self.btn1.grid(column=1,row=2+row_trasl,sticky="nsew",padx=padx_1,pady=pady_1)
        # create a Button for the selection of a setting file
        self.btn2 = Button(self.window, text='Select the setting file',command=self.select_setting_file)
        self.btn2.grid(column=0, row=3+row_trasl,sticky="nsew",padx=padx_0,pady=pady_0)
        # create  a Label where to show the selected setting file
        self.lbl2 = Label(self.window,text='',bg="white")
        self.lbl2.grid(column=1,row=3+row_trasl,sticky="nsew",padx=padx_1,pady=pady_1)
        # create a Button to select to select the data path
        self.btn3 = Button(self.window, text='Select path to data', command= self.select_path_to_data)
        self.btn3.grid(column=0,row=4+row_trasl,sticky="nsew",padx=padx_0,pady=pady_0)
        # create a Label to show the path to data selected
        self.lbl3 = Label(self.window,text='',bg="white")
        self.lbl3.grid(column=1,row=4+row_trasl,sticky="nsew",padx=padx_1,pady=pady_1) 
        
        # put a label for the feature type combobox
        self.labelTop = Label(self.window,text="Choose the feature kind")
        self.labelTop.grid(column=0,row=5+row_trasl,padx=padx_0,pady=pady_0,sticky="nsew")
        # instantiate a tuple for the string of the three modalities
        data=("Deep Features", "Traditional Features")
        # create the combobox for the modality selection
        self.cb_feat_kind=Combobox(self.window, values=data)
        self.cb_feat_kind.grid(column=0, row=6+row_trasl,sticky="nsew",padx=padx_0,pady=pady_0-5)
        # bind the selection to a callback function
        self.cb_feat_kind.bind("<<ComboboxSelected>>",self.callbackFunc_feat_kind)
        
        
        
        
        # create a label for the combobox to visualize image alterations
        self.lbl_cb3 = Label(self.window,text='Visualize Images alterations')
        self.lbl_cb3.grid(column=1, row=5+row_trasl, sticky='nsew',padx=padx_1,pady=pady_1)
        # create a combobox to to visualize image alterations
        self.cb3=Combobox(self.window)
        self.cb3.grid(column=1, row=6+row_trasl, sticky='nsew',padx=padx_1,pady=pady_1-5)
        # bind the combo to a callback function and disable its state until the dataset and parameters are selected
        self.cb3.bind("<<ComboboxSelected>>",self.visualize_images)
        self.cb3["state"] = 'disabled'
        # initialize a Boolean variable accounting for the saving option of normal and modified datasets
        self.var1 = False # set the Boolean variable to False
        # create a Checkbutton to save or not the normal and modified image datasets and place it
        self.check_btn = Checkbutton(self.window,text= 'Save normal and modified images?', command=self.saveimages,variable=self.var1)
        self.check_btn.grid(column=0,row=6+row_trasl+2,sticky="nsew",padx=padx_0,pady=pady_0)
        # create a button to save normal images (by default is disabled until self.var1== True)
        self.btn5 = Button(self.window,text='Select Path to normal images', command = self.save_normal_images, state=DISABLED)
        self.btn5.grid(column=0,row=7+row_trasl+2,sticky="nsew",padx=padx_0,pady=pady_0)
        # create a button to visualize the selected path to normal image dataset
        self.lbl5 = Label(self.window,text='',bg="Light Gray")
        self.lbl5.grid(column=1,row=7+row_trasl+2,sticky="nsew",padx=padx_1,pady=pady_1)
        # create a button to save modified images (by default is disabled until self.var1== True)
        self.btn6 = Button(self.window, text='Select Path to modified images', command= self.save_modified_images, state=DISABLED)
        self.btn6.grid(column=0,row=8+row_trasl+2,sticky="nsew",padx=padx_0,pady=pady_0)
        # create a button to visualize the selected path to modified image dataset
        self.lbl6 = Label(self.window,text='',bg="Light Gray")
        self.lbl6.grid(column=1,row=8+row_trasl+2,sticky="nsew",padx=padx_1,pady=pady_1)
        # create a button to perform tests
        self.btn7 = Button(self.window, text='Perform Tests', command= self.perform_test)
        self.btn7.grid(column=0,row=9+row_trasl+2, sticky="nsew",padx=padx_0,pady=pady_0)
        # disable its state until all requirements are satisfied
        self.btn7["state"] = DISABLED
        # set the underlying label to No test Performed
        self.lbl_1 = Label(self.window, text='No Tests Performed')
        self.lbl_1.grid(column=1, row= 9+row_trasl+2,padx=padx_1,pady=pady_1)
        
        # create a label for the combo box
        self.lbl_cb2 = Label(self.window,text='Visualize SENS vs DP scatter plots')
        self.lbl_cb2.grid(column=1, row=10+row_trasl+2, sticky='nsew',padx=padx_1,pady=0)
        # create a combobox to Visualize SENS vs DP scatter plots  
        self.cb2=Combobox(self.window)
        self.cb2.grid(column=1, row=11+row_trasl+2, sticky='nsew',padx=padx_1,pady=0)
        # bind the combo to a callback function
        self.cb2.bind("<<ComboboxSelected>>",self.callbackFunc2)
        # disable its state until tests are performed
        self.cb2["state"] = 'disabled'
        
        
        # create a button to save features and disable its state unless tests are performed
        self.btn8 = Button(self.window,text='Choose features',command=self.show_results)
        self.btn8.grid(column=0,row=13+row_trasl+2,sticky = 'nsew',padx=padx_0,pady=pady_0)
        self.btn8["state"] = DISABLED
        
        
        
        # create a button for the selection of the test images and disable it until features are selected
        self.btn11 = Button(self.window, text="Select folder containing test images", command=self.test_images,state = DISABLED)
        self.btn11.grid(column=0, row=14+row_trasl+2, sticky='nsew',padx=padx_0,pady=pady_0)
        # create a label to visualize the path to the selected test images
        self.lbl11 = Label(self.window, text="",bg="white")
        self.lbl11.grid(column=1, row=14+row_trasl+2, sticky='nsew',padx=padx_1,pady=pady_1)
        # initialize a check (Boolean variable) in order to enable the save button
        self.check1 = False
        # create Button to select the folder where to save results
        self.btn12 = Button(self.window, text="Select folder where to save results", command=self.select_folder_results, state = DISABLED)
        self.btn12.grid(column=0, row=15+row_trasl+2, sticky = 'nsew',padx=padx_0,pady=pady_0)
        # create the label to visualize the selected folder for results
        self.lbl12 = Label(self.window, text = "",bg="white")
        self.lbl12.grid(column=1,row=15+row_trasl+2,sticky='nsew',padx=padx_1,pady=pady_1)
        # iniatialize a check (Boolean variable) in order to enable the save button
        self.check2 = False
        # create a button to save results
        self.btn13 = Button(self.window, text="Save results",command=self.save_results,state=DISABLED)
        self.btn13.grid(column=0, row=16+row_trasl+2, sticky='nsew',padx=padx_0,pady=pady_0)
        
        
        # create a button to reset the GUI
        self.btn0 = Button(self.window,text='Reset GUI',command = self.reset_gui)
        self.btn0.grid(column=0,row=17+row_trasl+2,padx=padx_0,pady=80,sticky='nsew')
        # Import the image using PhotoImage function
        self.click_btn= PhotoImage(file='BEE.png')
        self.btn_web = Button(self.window, image=self.click_btn, command=self.openweb)
        self.btn_web.grid(column=1,row=17+row_trasl+2,sticky=SE,padx=padx_0,pady=40)
        # put the icon 
        self.window.iconbitmap(icon_name)
        
        # run the main loop
        self.window.mainloop()
    def select_folder_results(self):
        # this function is associated with self.btn12 clicking
        # takes no input arguments
        
        # get current working directory
        current_dir = os.getcwd()
        # ask the directory
        f = filedialog.askdirectory(title='Select the folder saving files',initialdir=current_dir)
        try:
            # if f is not empty store f inside the var_save_path attribute
            assert f!=''
            self.var_save_path = f
        except:
            # otherwise display a message box showing the warning that the assignation was not succesfully completed
            messagebox.showwarning(title='Warning', message='please select a folder')
            return
        # store inside th var_save_path2 variable the last name for visualization purposes
        var_save_path2 = f.split('/')
        var_save_path2 = var_save_path2[-1]
        # display it in lbl12 position
        self.lbl12.config(text=var_save_path2)
        # store the name of files where to save DM parameters inside var_DM_parameter_path attribute
        self.var_DM_parameter_path = os.path.sep.join([self.var_save_path,"DM_parameters.mat"])
        # store the name of files where to save the training dataset inside var_training_dataset_path attribute
        self.var_training_dataset_path = os.path.sep.join([self.var_save_path,"training_dataset.mat"])
        # store the name of files where to save the training dataset inside var_test_dataset_path attribute
        self.var_test_dataset_path = os.path.sep.join([self.var_save_path,"test_dataset.mat"])
        
        # set check2 attribute to True
        self.check2 = True
        
        # make a check in order to see if we can enable the save results button (btn13)
        if self.var1:
            if (self.check1) and (self.check2) and (self.var_normal_dataset_path!='') and (self.var_modified_dataset_path!=''):
                 self.btn13["state"] = NORMAL
            else:
                self.btn13["state"] = DISABLED
        else:
            if (self.check1)and(self.check2):
                self.btn13["state"] = NORMAL
            else:
                self.btn13["state"] = DISABLED
    def callbackFunc_feat_kind(self,event):
        # this calback function is binded to the selection of a modality anlysis (self.cb)
        
        # get the var_name_modality selected
        self.var_feature_modality = event.widget.get()
        # "Deep Features", "Traditional Features"
        # store the proper set of tuple containing strings coding for the implemented tests
        if self.var_feature_modality == "Deep Features":
            self.deep_features = 1;
            self.traditional_features = 0;
        if self.var_feature_modality == "Traditional Features":
            self.deep_features = 0;
            self.traditional_features = 1;
        
        if self.var1:
            # than var_normal_dataset_path and var_modified_dataset_path beyond var_name_modality, var_setting_file, and var_class_path have to be assigned to enable the perform test button
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!='') and (self.var_normal_dataset_path!='') and (self.var_modified_dataset_path!=''):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        else:
            # if normal and modified images do not have to be saved
            # than var_name_modality, var_setting_file, and var_class_path have to be assigned to enable the perform test button
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        #if var_name_modality, var_setting_file, and var_class_path are assigned than enable the visualize modified image combobox
        if (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                
                self.cb3["state"] = 'normal'
        else:
                
                self.cb3["state"] = 'disabled'
    
    def callbackFunc(self,event):
        # this calback function is binded to the selection of a modality anlysis (self.cb)
        
        # get the var_name_modality selected
        self.var_name_modality = event.widget.get()
        # store the proper set of tuple containing strings coding for the implemented tests
        if self.var_name_modality == '2D brightfield':
            data = ("luminance test", "movement test", "out-of-focus test")
        if self.var_name_modality == '3D phase-contrast':
            data = ("luminance test", "gel-pattern test", "out-of-focus test")
        if self.var_name_modality == '3D fluorescence':
            data = ("autofluorescence test","photobleaching test","saturation test")
        # store the generated tuple inside the values of the combobox cb3 used for plotting example images
        self.cb3["values"] = data
        # if normal and modified images have to be saved
        if self.var1:
            # than var_normal_dataset_path and var_modified_dataset_path beyond var_name_modality, var_setting_file, and var_class_path have to be assigned to enable the perform test button
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!='') and (self.var_normal_dataset_path!='') and (self.var_modified_dataset_path!=''):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        else:
            # if normal and modified images do not have to be saved
            # than var_name_modality, var_setting_file, and var_class_path have to be assigned to enable the perform test button
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        #if var_name_modality, var_setting_file, and var_class_path are assigned than enable the visualize modified image combobox
        if (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                
                self.cb3["state"] = 'normal'
        else:
                
                self.cb3["state"] = 'disabled'
    
    def openweb(self):
        # this function is used to open the web page of our research group
        webbrowser.open(self.url,new=self.new)
    
    def saveimages(self):
        # this method is linked to the check button used to enable saving the normal and modified images
        
        # first change the boolean attribute (when the GUI starts is setted to False)
        
        self.var1 = not(self.var1)
        # if var1 is True than enable buttons used to select 
        if self.var1 == True:
            self.btn5["state"] = NORMAL
            self.btn6["state"] = NORMAL
            self.lbl5.config(bg="white")
            self.lbl6.config(bg="white")
        else:
            self.btn5["state"] = DISABLED
            self.btn6["state"] = DISABLED
            
            if self.normal_images_path != '':
                self.normal_images_path = ''
            
            if self.modified_images_path != '':
                self.modified_images_path = ''
            self.lbl5.config(bg="Light Gray",text=self.normal_images_path)
            self.lbl6.config(bg="Light Gray",text=self.modified_images_path)    
        if self.var1:
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!='') and (self.var_normal_dataset_path) and (self.var_modified_dataset_path):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        else:
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        if (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                
                self.cb3["state"] = 'normal'
        else:
                
                self.cb3["state"] = 'disabled'
                
    def save_normal_images(self):
        # this method is linked to "Select Path to normal images" (btn5)
        # get current working directory
        current_dir = os.getcwd()
        # set default name
        name_file = 'dataset_normal_images'
        name = filedialog.asksaveasfilename(title='Choose the name and location for the file where to save the dataset of normal images',defaultextension='.mat',
                                                initialfile=name_file,initialdir=current_dir)
        try:
            
            assert name!=''
            self.var_normal_dataset_path = name
        except:
            # if name is empty show a warning
            messagebox.showwarning(title='Warning', message='please select a path')
            return
        # string for visualization 
        var_normal_dataset_path2 = name.split('/')
        var_normal_dataset_path2 = var_normal_dataset_path2[-1]
        self.lbl5.config(text=var_normal_dataset_path2)
        # choose whether or not to enable btn7 and cb3
        if self.var1:
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!='') and (self.var_normal_dataset_path) and (self.var_modified_dataset_path):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        else:
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        if (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                
                self.cb3["state"] = 'normal'
        else:
                
                self.cb3["state"] = 'disabled'
                

    def save_modified_images(self):
        # this method is linked to btn 6
        # get current working directory
        current_dir = os.getcwd()
        # set default name
        name_file = 'dataset_modified_images'
        name = filedialog.asksaveasfilename(title='Choose the name and location for the file where to save the dataset of modified images',defaultextension='.mat',
                                                initialfile=name_file,initialdir=current_dir)
        try:
            assert name!=''
            self.var_modified_dataset_path = name
        except:
            # if name is empty show a warning
            messagebox.showwarning(title='Warning', message='please select a path')
            return
        # string for visualization 
        var_modified_dataset_path2 = name.split('/')
        var_modified_dataset_path2 = var_modified_dataset_path2[-1]
        self.lbl6.config(text=var_modified_dataset_path2)
        # choose whether or not to enable btn7 and cb3
        if self.var1:
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!='') and (self.var_normal_dataset_path) and (self.var_modified_dataset_path):
                self.btn7["state"] = NORMAL
                
                
            else:
                self.btn7["state"] = DISABLED
                
        else:
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
               
        if (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                
                self.cb3["state"] = 'normal'
        else:
                
                self.cb3["state"] = 'disabled'
                
        

    def select_path_to_data(self):
        # this method is linked to btn3
        # get current working folder
        current_dir = os.getcwd()
        # ask directory
        f = filedialog.askdirectory(title='Select the folder containing subdirectories with images of the two classes',initialdir=current_dir)
        try:
            assert f!=''
            self.var_class_path = f
        except:
            # id f is empty show a warning
            messagebox.showwarning(title='Warning', message='please select a folder')
            return
        # string for visualization
        var_class_path2 = f.split('/')
        var_class_path2 = var_class_path2[-1]
        self.lbl3.config(text=var_class_path2)
        # choose whether or not to enable btn7 and cb3
        if self.var1:
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!='') and (self.var_normal_dataset_path) and (self.var_modified_dataset_path):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        else:
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                self.btn7["state"] = NORMAL
            else:
                self.btn7["state"] = DISABLED
                
        if (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                
                self.cb3["state"] = 'normal'
        else:
                
                self.cb3["state"] = 'disabled'
                

    
    def select_setting_file(self):
        # this method is linked to btn2
        # get current working directory
        current_dir = os.getcwd()
        # ask to open a file 
        f = filedialog.askopenfilename(title='Select the file txt with instructions',initialdir=current_dir)
        try:
            assert f!='' 
            self.var_setting_file = f
        except:
            # if f is empty show a warning
            tkinter.messagebox.showwarning(title='Warning',message='please select a file')
            
            return
        # string for visualization
        var_setting_file2 = f.split('/')
        var_setting_file2 = var_setting_file2[-1]
        self.lbl2.config(text=var_setting_file2)
        self.btn1["state"] = NORMAL
        # choose whether or not to enable btn7 and cb3
        if self.var1:
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!='') and (self.var_normal_dataset_path) and (self.var_modified_dataset_path):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        else:
            if (self.deep_features!='') and (self.traditional_features!='') and (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                self.btn7["state"] = NORMAL
                
            else:
                self.btn7["state"] = DISABLED
                
        if (self.var_name_modality!='') and (self.var_setting_file!='') and (self.var_class_path!=''):
                
                self.cb3["state"] = 'normal'
        else:
                
                self.cb3["state"] = 'disabled'
        # read the setting file
        self.read_only_setting_file()
    def read_only_setting_file(self):
        # this method is called at the end of the previous one select_setting_file and enables reading the setting file
        # open the txt file
        f = open(self.var_setting_file,mode='r')
        # read a line
        line = f.readline()
        # initialize a counter and a dictionary
        tline = {}
        n = 0
        
        while line!='':
          
            if n%2==0:
                name_key = line.split()
            else:
                # construct a dictionary taking keys from pair rows and values from odd rows
                tline[name_key[0]] = line.split()[0]
    
            line = f.readline()
            n += 1
    
        keys = tline.keys()
        # store the values contained in the setting file inside the corresponding attributes
        for name in keys: 
            if name == '%net':
                net_name = tline[name]
                self.var_net_name = net_name
                
                
            if name == '%layer':
                layer = tline[name]
                
                self.var_layer_name = layer
                
                
            if name == '%lev_bright':
                string = tline[name]
                
                self.var_lev_bright = string
                
                del string
            if name == '%lev_rot':
                string = tline[name]
                
                self.var_lev_rot = string
                
                del string
                
            if name == '%lev_trasl':
                string = tline[name]
                
                self.var_lev_trasl = string
                
                del string
            if name == '%lev_focus':
                string = tline[name]
                
                self.var_lev_focus = string
                
                del string
                
            if name == '%lambda_pattern':
                string = tline[name]
                
                self.var_lev_lambda_pattern = string
                
                del string
            if name == '%perc_bleach':
                string = tline[name]
                
                self.var_lev_perc_bleach = string
                
                del string
            if name == '%th_fluo':
                string = tline[name]
                
                self.var_lev_th_fluo = string
                
                del string
                
            if name == '%th_DP':
                string = tline[name]
                
                self.var_th_DP = string
                
                del string
                
            if name == '%th_SENS':
                string = tline[name]
                
                self.var_th_SENS = string
                
                del string
    def read_setting_file(self):
        # this method is called by make_new_window method which is linked to btn1 ("Change the provided settings")
        
        # first open the file containing the setting informations 
        # this setting will be used only for the keys 
        f = open(self.var_setting_file,mode='r')
        # read the first line
        line = f.readline()
        # initialize an empty dictionary and counter
        tline = {}
        n = 0
        
        # construct a dictionary taking keys from pair rows and values from odd rows
        while line!='':
          
            if n%2==0:
                name_key = line.split()
            else:
                tline[name_key[0]] = line.split()[0]
    
            line = f.readline()
            n += 1
        pady_0 = 5
        col_grid = 0 # column for the Entries
        keys = tline.keys()
        i = 1;
        for name in keys: 
            if name == '%net':
                # initialize String variable
                self.var_net_name_ = StringVar()
                # set the value to the one read frome the file
                self.var_net_name_.set(self.var_net_name)
                # write the label for the entry
                self.label_net_name = Label(self.window2,text="name of the selected network")
                # place the label inside the window
                self.label_net_name.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # Initialize the entry and the value displayed to the one read from the setting file
                self.net_name_widget = Entry(self.window2,textvariable=self.var_net_name_)
                self.net_name_widget.insert(0,self.var_net_name_.get())
                # update counter
                i += 1;
                # place the entry inside the window
                self.net_name_widget.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
            if name == '%layer':
                # initialize String variable
                self.var_layer_name_ = StringVar()
                # set the value to the one read frome the file
                self.var_layer_name_.set(self.var_layer_name)
                # write the label for the entry
                self.label_layer_name = Label(self.window2,text="name of the selected layer")
                # place the label inside the window
                self.label_layer_name.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
                # Initialize the entry and the value displayed to the one read from the setting file
                self.layer_name_widget = Entry(self.window2,textvariable=self.var_layer_name_.get())
                self.layer_name_widget.insert(0,self.var_layer_name_.get())
                # place the entry inside the window
                self.layer_name_widget.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
            if name == '%lev_bright':
                # initialize String variable
                self.var_lev_bright_ = StringVar()
                # set the value to the one read frome the file
                self.var_lev_bright_.set(self.var_lev_bright)
                # write the label for the entry
                self.label_lev_bright = Label(self.window2,text="range of brightness shift")
                # place the label inside the window
                self.label_lev_bright.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
                # Initialize the entry and the value displayed to the one read from the setting file
                self.lev_bright_widget = Entry(self.window2,textvariable=self.var_lev_bright_)
                self.lev_bright_widget.insert(0,self.var_lev_bright_.get())
                # place the entry inside the window
                self.lev_bright_widget.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
                
            if name == '%lev_rot':
                # initialize String variable
                self.var_lev_rot_ = StringVar()
                # set the value to the one read frome the file
                self.var_lev_rot_.set(self.var_lev_rot)
                # write the label for the entry
                self.label_lev_rot = Label(self.window2,text="range of rotation shift")
                # place the label inside the window
                self.label_lev_rot.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
                # Initialize the entry and the value displayed to the one read from the setting file
                self.lev_rot_widget = Entry(self.window2,textvariable=self.var_lev_rot_)
                self.lev_rot_widget.insert(0,self.var_lev_rot_.get())
                # place the entry inside the window
                self.lev_rot_widget.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
            if name == '%lev_trasl':
                # initialize String variable                
                self.var_lev_trasl_ = StringVar()
                # set the value to the one read frome the file
                self.var_lev_trasl_.set(self.var_lev_trasl)
                # write the label for the entry
                self.label_lev_trasl = Label(self.window2,text="range of traslation shift")
                # place the label inside the window
                self.label_lev_trasl.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
                # Initialize the entry and the value displayed to the one read from the setting file
                self.lev_trasl_widget = Entry(self.window2,textvariable=self.var_lev_trasl_)
                self.lev_trasl_widget.insert(0,self.var_lev_trasl_.get())
                # place the entry inside the window
                self.lev_trasl_widget.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
            if name == '%lev_focus':
                # initialize String variable 
                self.var_lev_focus_ = StringVar()
                # set the value to the one read frome the file
                self.var_lev_focus_.set(self.var_lev_focus)
                # write the label for the entry
                self.label_lev_focus = Label(self.window2,text="range of focus shift")
                # place the label inside the window
                self.label_lev_focus.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
                # Initialize the entry and the value displayed to the one read from the setting file
                self.lev_focus_widget = Entry(self.window2,textvariable=self.var_lev_focus_)
                self.lev_focus_widget.insert(0,self.var_lev_focus_.get())
                # place the entry inside the window
                self.lev_focus_widget.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
            if name == '%lambda_pattern':
                # initialize String variable 
                self.var_lev_lambda_pattern_ = StringVar()
                # set the value to the one read frome the file
                self.var_lev_lambda_pattern_.set(self.var_lev_lambda_pattern)
                # write the label for the entry
                self.label_lev_lambda_pattern = Label(self.window2, text="range of lambda pattern")
                # place the label inside the window
                self.label_lev_lambda_pattern.grid(column=col_grid, row = i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1
                # Initialize the entry and the value displayed to the one read from the setting file
                self.lev_lambda_pattern_widget = Entry(self.window2, textvariable=self.var_lev_lambda_pattern_)
                self.lev_lambda_pattern_widget.insert(0,self.var_lev_lambda_pattern_.get())
                # place the entry inside the window
                self.lev_lambda_pattern_widget.grid(column=col_grid, row = i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1
            if name == '%perc_bleach':
                # initialize String variable               
                self.var_lev_perc_bleach_ = StringVar()
                # set the value to the one read frome the file
                self.var_lev_perc_bleach_.set(self.var_lev_perc_bleach)
                # write the label for the entry
                self.label_lev_perc_bleach = Label(self.window2, text="range of perch_bleach pattern")
                # place the label inside the window
                self.label_lev_perc_bleach.grid(column=col_grid, row = i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1
                # Initialize the entry and the value displayed to the one read from the setting file
                self.lev_perc_bleach_widget = Entry(self.window2, textvariable=self.var_lev_perc_bleach_)
                self.lev_perc_bleach_widget.insert(0,self.var_lev_perc_bleach_.get())
                # place the entry inside the window
                self.lev_perc_bleach_widget.grid(column=col_grid, row = i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1
                
            if name == '%th_fluo':
                # initialize String variable
                self.var_lev_th_fluo_ = StringVar()
                # set the value to the one read frome the file
                self.var_lev_th_fluo_.set(self.var_lev_th_fluo)
                # write the label for the entry
                self.label_lev_th_fluo = Label(self.window2, text="range of th_fluo pattern")
                # place the label inside the window
                self.label_lev_th_fluo.grid(column=col_grid, row = i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1
                # Initialize the entry and the value displayed to the one read from the setting file
                self.lev_th_fluo_widget = Entry(self.window2, textvariable=self.var_lev_th_fluo_)
                self.lev_th_fluo_widget.insert(0,self.var_lev_perc_bleach_.get())
                # place the entry inside the window
                self.lev_th_fluo_widget.grid(column=col_grid, row = i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1
            if name == '%th_DP':
                # initialize String variable
                self.var_th_DP_ = StringVar()
                # set the value to the one read frome the file
                self.var_th_DP_.set(self.var_th_DP)
                # write the label for the entry
                self.label_th_DP = Label(self.window2,text="Value of the DP threshold")
                # place the label inside the window
                self.label_th_DP.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
                # Initialize the entry and the value displayed to the one read from the setting file
                self.th_DP_widget = Entry(self.window2,textvariable=self.var_th_DP_)
                self.th_DP_widget.insert(0,self.var_th_DP_.get())
                # place the entry inside the window
                self.th_DP_widget.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;                
            if name == '%th_SENS':
                # initialize String variable                
                self.var_th_SENS_ = StringVar()
                # set the value to the one read frome the file
                self.var_th_SENS_.set(self.var_th_SENS)
                # write the label for the entry
                self.label_th_SENS = Label(self.window2,text="Value of the SENS threshold")
                # place the label inside the window
                self.label_th_SENS.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
                # Initialize the entry and the value displayed to the one read from the setting file
                self.th_SENS_widget = Entry(self.window2,textvariable=self.var_th_SENS_)
                self.th_SENS_widget.insert(0,self.var_th_SENS_.get())
                # place the entry inside the window
                self.th_SENS_widget.grid(column=col_grid, row=i, sticky='nsew',padx = padx_0,pady=pady_0)
                # update counter
                i += 1;
    def reset_gui(self):
        # this method is linked to the reset GUI button
        # first destroy the window 
        self.window.destroy()
        # then set the default again
        self.set_gui()
    def use_these_settings(self):
        # this method is linked to the Update Settings button of the window created by the make_new_window method
        # the idea is to get what is written inside the Entries and copy it inside the attributes that will be successively used
        
        # th_SENS and th_DP need no check for existence because they are always present for the method
        self.var_th_SENS = self.th_SENS_widget.get()
        self.var_th_DP = self.th_DP_widget.get()
        # the other parameters are linked to tests specific for the application
        # hence we have to first check their existence
        if self.var_lev_th_fluo_ != None:
            self.var_lev_th_fluo = self.lev_th_fluo_widget.get()
        if self.var_lev_perc_bleach_ != None:
            self.var_lev_perc_bleach = self.lev_perc_bleach_widget.get()
        if self.var_lev_lambda_pattern_ != None:
            self.var_lev_lambda_pattern = self.lev_lambda_pattern_widget.get()
        if self.var_lev_focus_ != None:
            self.var_lev_focus = self.lev_focus_widget.get()
        if self.var_lev_trasl_ != None:
            self.var_lev_trasl = self.lev_trasl_widget.get()
        if self.var_lev_rot_ != None:
            self.var_lev_rot = self.lev_rot_widget.get()
        if self.var_lev_bright_ != None:
            self.var_lev_bright = self.lev_bright_widget.get()
        if self.var_layer_name_ != None:
            self.var_layer_name = self.layer_name_widget.get()
        if self.var_net_name_ != None:
            self.var_net_name = self.net_name_widget.get()
    def use_these_settings_and_close_window(self):
        # this function is called by btn_1 of the make_new_window
        # first use the settings
        self.use_these_settings()
        # then close the window
        self.window2.destroy()
    def make_new_window(self):
        # this method is linked to bnt1 
        # create a new window
        self.window2 = Tk()
        # title it Change settings
        self.window2.title('Change settings')
        # define a geometry
        self.window2.geometry("700x700+10+20")
        # define the column and row where to put button
        col_grid = 1
        row_count = 1
        # use the read setting file method
        self.read_setting_file()
        # create a button to update the settings
        self.btn_1 = Button(self.window2,text = "Update settings", command = self.use_these_settings_and_close_window)
        # place it in the window
        self.btn_1.grid(column = col_grid, row= row_count, sticky="nsew", padx = padx_0,pady=pady_0)
        # balance the weight for graphic visualization
        for r  in range(18):
            Grid.rowconfigure(self.window2, r, weight=1, minsize=30)
        for c in range(2):
            Grid.columnconfigure(self.window2, c, weight=1)
        # put the icon
        self.window2.iconbitmap(icon_name)
        # loop over the window
        self.window2.mainloop()
        
    def visualize_images(self,event):
        # this method is linked to the cb3 combobox and it enables image visualization
        # first get the selected test
        self.var_test_selected = event.widget.get()
        # create a list for images from the input images
        input_paths = list(paths.list_images(self.var_class_path))
        # create a list containing only the labels (corresponding to the second last folder)
        labels = [p.split(os.path.sep)[-2] for p in input_paths]
        # instantiate a member of label encoder class
        le = LabelEncoder()
        # create the vector containing the two class
        labels = le.fit_transform(labels)
        # create id for class 1
        cl0 = np.where(labels==0)[0]
        # create id for class 2
        cl1 = np.where(labels==1)[0]
        # select 1 elements per class
        id0 = np.random.randint(0,len(cl0),size=1)
        id1 = np.random.randint(0,len(cl1),size=1)
        # select the corresponding id
        cl0 = cl0[id0]
        cl1 = cl1[id1]
        # select the corresponding input paths
        input_paths0 = []
        input_paths1 = []
        for i,j in zip(cl0,cl1):
            input_paths0.append(input_paths[i])
            input_paths1.append(input_paths[j])
            
        # select a approx dimension for visualization (it does not have to match the exact input dimensions of the net)
        fac = (224,224)
        # initialize empty list for images
        imds = []
        # loop over images of class 1
        for i, input_path0 in enumerate(input_paths0):
            # read and resize the image corresponding to the provided path
            im = my_read_resize(input_path0, fac)
            # append the image to the list
            imds.append(im)
            del im
        # convert the list in numpy array
        imds0_im = np.array(imds)
        # initialize empty list for images
        imds = []
        # loop over images of class 2
        for i, input_path1 in enumerate(input_paths1):
            # read and resize the image corresponding to the provided path
            im = my_read_resize(input_path1, fac)
            # append the image to the list
            imds.append(im)
            del im
        # convert the list in numpy array
        imds1_im = np.array(imds)
        # check which test must be performed
        if self.var_test_selected == 'luminance test':
            # read the value and transform it into a numpy array
            string = self.var_lev_bright
            string = string.split('[')[-1]
            string = string.split(']')[0]
            lista = string.split(':')  
            del string
            start = float(lista[0])
            step = float(lista[1])
            stop = float(lista[2])+step
            lev_bright = np.arange(start,stop, step)
            # apply the tests over the images of the two classes
            imds0_mod = test_luminance_lamp(lev_bright, input_paths0, fac,vis=False)
            imds1_mod = test_luminance_lamp(lev_bright, input_paths1, fac,vis=False)
        if self.var_test_selected == 'movement test':
            # read the value of rotations and transform it into a numpy array
            string = self.var_lev_rot
            string = string.split('[')[-1]
            string = string.split(']')[0]
            lista = string.split(':')   
            del string
            start = float(lista[0])
            step = float(lista[1])
            stop = float(lista[2])+step
            lev_rot = np.arange(start,stop, step)
            # read the value of traslations and transform it into a numpy array
            string = self.var_lev_trasl
            string = string.split('[')[-1]
            string = string.split(']')[0]
            lista = string.split(':')   
            del string
            start = float(lista[0])
            step = float(lista[1])
            stop = float(lista[2])+step
            lev_trasl = np.arange(start,stop, step)
            # apply the tests over the images of the two classes
            imds0_mod = test_movement(lev_rot,lev_trasl,input_paths0, fac,vis=False)
            imds1_mod = test_movement(lev_rot,lev_trasl,input_paths1, fac,vis=False)
        if self.var_test_selected == "out-of-focus test":
            # read the value of focus alteration and transform it into a numpy array
            string = self.var_lev_focus
            string = string.split('[')[-1]
            string = string.split(']')[0]
            lista = string.split(':')   
            del string
            start = float(lista[0])
            step = float(lista[1])
            stop = float(lista[2])+step
            lev_focus = np.arange(start,stop, step)
            # apply the tests over the images of the two classes
            imds0_mod = test_out_of_focus(lev_focus, input_paths0, fac,vis=False)
            imds1_mod = test_out_of_focus(lev_focus, input_paths1, fac,vis=False)
            
        if self.var_test_selected == "autofluorescence test":
            # read the value of autfluorescence alteration and transform it into a numpy array
            string = self.var_lev_bright
            string = string.split('[')[-1]
            string = string.split(']')[0]
            lista = string.split(':') 
            del string
            start = float(lista[0])
            step = float(lista[1])
            stop = float(lista[2])+step
            lev_bright = np.arange(start,stop, step)
            # apply the tests over the images of the two classes
            imds0_mod = test_autofluo_luminance_lamp(lev_bright, input_paths0, fac,vis=False)
            imds1_mod = test_autofluo_luminance_lamp(lev_bright, input_paths1, fac,vis=False)
        if self.var_test_selected == "photobleaching test":
            # read the value of photobleaching test and transform it into a numpy array
            string = self.var_lev_perc_bleach
            string = string.split('[')[-1]
            string = string.split(']')[0]
            lista = string.split(':')        
            del string
            start = float(lista[0])
            step = float(lista[1])
            stop = float(lista[2])+step
            lev_perc_bleach = np.arange(start,stop, step)
            # apply the tests over the images of the two classes
            imds0_mod = test_photobleaching(lev_perc_bleach,input_paths0, fac,vis=False)
            imds1_mod = test_photobleaching(lev_perc_bleach,input_paths1, fac,vis=False)
        if self.var_test_selected == "saturation test":
            # read the value of saturation test and transform it into a numpy array
            string = self.var_lev_th_fluo
            string = string.split('[')[-1]
            string = string.split(']')[0]
            lista = string.split(':')        
            del string
            start = float(lista[0])
            step = float(lista[1])
            stop = float(lista[2])+step
            lev_th_fluo = np.arange(start,stop, step)
            # apply the tests over the images of the two classes
            imds0_mod = test_saturation(lev_th_fluo, input_paths0, fac,vis=False)
            imds1_mod = test_saturation(lev_th_fluo, input_paths1, fac,vis=False)
        if self.var_test_selected == "gel-pattern test":
            # read the value of gel-pattern rotation alteration and transform it into a numpy array
            string = self.var_lev_rot
            string = string.split('[')[-1]
            string = string.split(']')[0]
            lista = string.split(':')
            del string
            start = float(lista[0])
            step = float(lista[1])
            stop = float(lista[2])+step
            lev_rot = np.arange(start,stop, step)
            # read the value of gel-pattern lambda alteration and transform it into a numpy array
            string = self.var_lev_lambda_pattern
            string = string.split('[')[-1]
            string = string.split(']')[0]
            lista = string.split(':')        
            del string
            start = float(lista[0])
            step = float(lista[1])
            stop = float(lista[2])+step
            lev_lambda_pattern = np.arange(start,stop, step)
            # apply the tests over the images of the two classes
            imds0_mod = test_gel_pattern(lev_rot,lev_lambda_pattern,input_paths0, fac,vis=False)
            imds1_mod = test_gel_pattern(lev_rot,lev_lambda_pattern,input_paths1, fac,vis=False)
        # open a new window
        win2 = Tk() 
        # Set a window title
        win2.title("Example of images after "+self.var_test_selected)
        # set a window geometry
        win2.geometry("700x700+10+20")
        # make a plot with matplotlib
        fig = plt.figure(figsize=(6,8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
              
        # set title for subplots
        ax1.title.set_text('Cell class 1 normal')
        ax2.title.set_text('Cell class 1 modified')
        ax3.title.set_text('Cell class 2 normal')
        ax4.title.set_text('Cell class 2 modified')
        
        # show corresponding images
        ax1.imshow(imds0_im[0])
        ax2.imshow(imds0_mod[0])
        ax3.imshow(imds1_im[0])
        ax4.imshow(imds1_mod[0])
        # use FigureCanvasTkAgg in order to display images on the window
        canvas = FigureCanvasTkAgg(fig, master=win2)
        canvas.get_tk_widget().pack(fill=tkinter.BOTH,expand=1)
        canvas.draw()
        # set the icon 
        win2.iconbitmap(icon_name)
        # mainloop over the selected window
        win2.mainloop()
    
    def save_image_utils(self, a, b, id1, id2, id3, id_, DP0, th_DP, th_SENS,var_test_selected):
        # this method is called by the callbackFunc2method
        
        # open a new window
        win3 = Tk()
        # set window title
        win3.title("SENSvsDP score plot")
        # open a new figure and plot starting Discriminative powers
        fig = plt.figure()
        plt.plot(np.zeros((len(id_))),DP0[id_],'sr',markerfacecolor='r', label='features selected based only on original dataset')
        # plot selected features in cyan
        plt.plot(b[id1],a[id1],'sc',markerfacecolor='c',label='Selected features')
        # plot features with too high Sensitivity in blue
        plt.plot(b[id2],a[id2],'sb',markerfacecolor='b',label='Too high sesitivity')
        # plot features with too low DP
        plt.plot(b[id3],a[id3],'sg',markerfacecolor='g',label='Too low DP in the modified dataset')
        # plot axis names
        plt.xlabel('sensitivity to ' + var_test_selected, fontsize=14)
        plt.ylabel('DP',fontsize=14)
        # plot a line that represents the threshold of DP above which
        # the features are selected
        x_values1 = np.array([0, np.max(b)])
        y_values1 = np.array([th_DP, th_DP])
        
        plt.plot(x_values1, y_values1,color=[0.93,0.69,0.13],linewidth=2,linestyle='--')
        # plot a line that represents the threshold of Sensitivity
        # below which the features are selected 
        x_values2 = np.array([th_SENS,th_SENS])
        y_values2 = np.array([0.5, np.max(a)])
        plt.plot(x_values2,y_values2, color=[0.93,0.69,0.13],linewidth=2,linestyle='--')
        plt.legend()
        # use FigureCanvasTkAgg to plot in window
        canvas = FigureCanvasTkAgg(fig, master=win3)
        canvas.get_tk_widget().pack(fill=tkinter.BOTH,expand=1)
        canvas.draw()
        # put icon 
        win3.iconbitmap(icon_name)
        # save figure
        plt.savefig("SENSvsDP "+var_test_selected+".png")
        # main loop over the window
        win3.mainloop()
        
    
    def callbackFunc2(self, event):
        # this method is linked to cb2 combobox used to display SENS-DP plots
        
        # get the event
        self.var_test_selected = event.widget.get()
        
        if self.var_test_selected == 'luminance test':
            # get the necessary variables for the plots
            a = self.a_luminance
            b = self.b_luminance
            id1 = self.id1_luminance
            id2 = self.id2_luminance
            id3 = self.id3_luminance
            id_ = self.id_luminance
            DP0 = self.DP0_luminance
            th_DP = self.th_DP_luminance
            th_SENS = self.th_SENS_luminance
            # call the method used for plotting
            self.save_image_utils(a, b, id1, id2, id3, id_, DP0, th_DP, th_SENS, self.var_test_selected)
            
        if self.var_test_selected == 'movement test':
            # get the necessary variables for the plots
            a = self.a_movement
            b = self.b_movement
            id1 = self.id1_movement
            id2 = self.id2_movement
            id3 = self.id3_movement
            id_ = self.id_movement
            DP0 = self.DP0_movement
            th_DP = self.th_DP_movement
            th_SENS = self.th_SENS_movement
            # call the method used for plotting
            self.save_image_utils(a, b, id1, id2, id3, id_, DP0, th_DP, th_SENS, self.var_test_selected)
            
        if  self.var_test_selected == "out-of-focus test":
            # get the necessary variables for the plots
            a = self.a_out_of_focus
            b = self.b_out_of_focus
            id1 = self.id1_out_of_focus
            id2 = self.id2_out_of_focus
            id3 = self.id3_out_of_focus
            id_ = self.id_out_of_focus
            DP0 = self.DP0_out_of_focus
            th_DP = self.th_DP_out_of_focus
            th_SENS = self.th_SENS_out_of_focus
            # call the method used for plotting
            self.save_image_utils(a, b, id1, id2, id3, id_, DP0, th_DP, th_SENS, self.var_test_selected)
            
        if self.var_test_selected == "autofluorescence test":
            # get the necessary variables for the plots
            a = self.a_autofluo
            b = self.b_autofluo
            id1 = self.id1_autofluo
            id2 = self.id2_autofluo
            id3 = self.id3_autofluo
            id_ = self.id_autofluo
            DP0 = self.DP0_autofluo
            th_DP = self.th_DP_autofluo
            th_SENS = self.th_SENS_autofluo
            # call the method used for plotting
            self.save_image_utils(a, b, id1, id2, id3, id_, DP0, th_DP, th_SENS, self.var_test_selected)
        if self.var_test_selected == "photobleaching test":
            # get the necessary variables for the plots
            a = self.a_photobleach
            b = self.b_photobleach
            id1 = self.id1_photobleach
            id2 = self.id2_photobleach
            id3 = self.id3_photobleach
            id_ = self.id_photobleach
            DP0 = self.DP0_photobleach
            th_DP = self.th_DP_photobleach
            th_SENS = self.th_SENS_photobleach
            # call the method used for plotting
            self.save_image_utils(a, b, id1, id2, id3, id_, DP0, th_DP, th_SENS, self.var_test_selected)
        if self.var_test_selected == "saturation test":
            # get the necessary variables for the plots
            a = self.a_saturation
            b = self.b_saturation
            id1 = self.id1_saturation
            id2 = self.id2_saturation
            id3 = self.id3_saturation
            id_ = self.id_saturation
            DP0 = self.DP0_saturation
            th_DP = self.th_DP_saturation
            th_SENS = self.th_SENS_saturation
            # call the method used for plotting
            self.save_image_utils(a, b, id1, id2, id3, id_, DP0, th_DP, th_SENS, self.var_test_selected)
        if self.var_test_selected == "gel-pattern test":
            # get the necessary variables for the plots
            a = self.a_gel_pattern
            b = self.b_gel_pattern
            id1 = self.id1_gel_pattern
            id2 = self.id2_gel_pattern
            id3 = self.id3_gel_pattern
            id_ = self.id_gel_pattern
            DP0 = self.DP0_gel_pattern
            th_DP = self.th_DP_gel_pattern
            th_SENS = self.th_SENS_gel_pattern
            # call the method used for plotting
            self.save_image_utils(a, b, id1, id2, id3, id_, DP0, th_DP, th_SENS, self.var_test_selected) 
        
            


    def perform_test(self):
        # this method is linked to btn7 and performs all the tests
        
        # disable the other buttons
        self.btn0["state"] = DISABLED
        self.cb["state"] = 'disabled'
        self.btn1["state"] = DISABLED
        self.btn2["state"] = DISABLED
        self.btn3["state"] = DISABLED
        self.cb3["state"]  = 'disabled'
        self.check_btn["state"] = 'disabled'
        self.btn7["state"] = DISABLED
        self.cb_feat_kind["state"] = 'disabled'
        # apply tests according to the chosen modality
        if self.var_name_modality=='2D brightfield':
            self.brightfield_tests()
            # insert the appropriate values to the combobox used for visualization
            self.cb2["values"] = ("luminance test", "movement test", "out-of-focus test")
        elif self.var_name_modality =='3D phase-contrast':
            self.phase_contrast_tests()
            # insert the appropriate values to the combobox used for visualization
            self.cb2["values"] = ("luminance test", "gel-pattern test", "out-of-focus test")
        elif self.var_name_modality=='3D fluorescence':
            self.fluorescence_tests()
            # insert the appropriate values to the combobox used for visualization
            self.cb2["values"] = ("autofluorescence test","photobleaching test","saturation test")
        
        # insert the string saying all tests were performed
        self.lbl_1.config(text='All tests were performed')
        
        # re enable all buttons
        self.btn8["state"] = NORMAL
        
        self.cb2["state"] = 'normal'
        
        self.btn0["state"] = NORMAL
        self.cb["state"] = 'normal'
        self.btn1["state"] = NORMAL
        self.btn2["state"] = NORMAL
        self.btn3["state"] = NORMAL
        self.cb3["state"]  = 'normal'
        self.check_btn["state"] = 'normal'
        self.btn7["state"] = NORMAL
        self.cb_feat_kind["state"] = 'normal'

    def fluorescence_tests(self):
        # instantiate network
        net = select_network(self.var_net_name)
        # instantiate layer
        layer = self.var_layer_name
        # select dimensions
        fac = net.layers[0].input.shape[1:3]
        # take string for lev_bright  and transform it in numpy array
        string = self.var_lev_bright
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_bright = np.arange(start,stop, step)
        del string
        # take string for th_fluo  and transform it in numpy array
        string = self.var_lev_th_fluo
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_th_fluo = np.arange(start,stop, step)
        del string
        # take string for perc_bleach  and transform it in numpy array
        string = self.var_lev_perc_bleach
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_perc_bleach = np.arange(start,stop, step)
        del string
        # take string for th_DP and th_SENS  and transform it in float numbers
        th_DP = float(self.var_th_DP)
        th_SENS = float(self.var_th_SENS)
        # create a transfer learning model with the selected layer and network
        model = construct_transfer_learning_model(net, layer)
        # select the path to the images
        file_path = self.var_class_path
        # store the first image dataset paths
        imds = list(paths.list_images(file_path)) #list of the path to the images
        # read all images and corresponding labels
        imds_im, GT = read_all(imds, fac) #images as numpy arrays
        # if images are intended to be saved save them with their corresponding labels
        if self.var1:
            print('Saving normal image dataset')
            mat = {}
            mat["images_class"] = imds_im
            mat["GT"] = GT
            savemat(self.var_normal_dataset_path,mat)
            del mat
        # check if deep or traditional features have to be extracted
        if (self.deep_features==1) and (self.traditional_features==0):
            # extract features from normal images
            print('Calculating Deep Feature dataset original...')
            self.feat = model.predict(imds_im)
        elif (self.deep_features==0) and (self.traditional_features==1):
            # extract features from normal images
            print('Calculating Traditional Feature dataset original...')
            self.feat = extract_standard_descriptors(imds_im)
            
        # store Ground truth labels in GT attribute
        self.GT = GT
        # loop over the available tests
        for ind in range(3):
            # label for visualization
            var_process = "Performing test " + str(ind+1) + " of 3"
            self.lbl_1.config(text=var_process)
            if ind==0: #test brightness
                # perform the autofluorescence test and result a set of modified images
                print('Applying 3D phase-contrast fluo autofluorescence test to image dataset...')
                imds_mod = test_autofluo_luminance_lamp(lev_bright, imds, fac)
                # if modified images are intended to be saved save them with their corresponding labels
                if self.var1:
                    mat = {}
                    mat["images_class"] = imds_mod
                    mat["GT"] = GT
                    lista = self.var_modified_dataset_path.split('.')
                    # add the test performed to the name
                    name_file = lista[0]+'_autofluorescence_test.'+lista[-1]
                    savemat(name_file,mat)
                    del mat
                # check if deep or traditional features have to be extracted   
                if (self.deep_features==1) and (self.traditional_features==0):
                    # extract deep features from the modified dataset    
                    print('Calculating Deep Feature modified by autofluorescence test...')
                    feat_mod = model.predict(imds_mod)
                elif (self.deep_features==0) and (self.traditional_features==1):
                    # extract deep features from the modified dataset
                    print('Calculating Traditional Feature modified by autofluorescence test...')
                    feat_mod = extract_standard_descriptors(imds_mod)   
                # compute the feature selection: here there is the heart of the algorithm (inside utils file)
                print('Calculating DP and sensitivity for feature selection...')
                feat_sel, SENS, DP_mod, DP_0, dict_for_plot = my_feature_selection(self.feat, feat_mod, GT, th_DP, th_SENS, ind)
                # store the id of the selected features inside the feat_sel_autofluorescence variable
                self.feat_sel_autofluorescence = feat_sel
                # store utility variables useful for SENS-DP scatter plots
                self.a_autofluo = dict_for_plot["a"]
                self.b_autofluo = dict_for_plot["b"]
                self.id1_autofluo = dict_for_plot["id1"]
                self.id2_autofluo = dict_for_plot["id2"]
                self.id3_autofluo = dict_for_plot["id3"]
                self.id_autofluo = dict_for_plot["id_"]
                self.DP0_autofluo = dict_for_plot["DP0"]
                self.th_DP_autofluo = dict_for_plot["th_DP"]
                self.th_SENS_autofluo = dict_for_plot["th_SENS"]
                
            if ind==1: #test movement
                # perform the movement test and result a set of modified images
                print('Applying 3D phase-contrast fluo photobleaching test to image dataset...')
                imds_mod = test_photobleaching(lev_perc_bleach,imds, fac)
                # if modified images are intended to be saved save them with their corresponding labels
                if self.var1:
                    mat = {}
                    mat["images_class"] = imds_mod
                    mat["GT"] = GT
                    lista = self.var_modified_dataset_path.split('.')
                    # add the test performed to the name
                    name_file = lista[0]+'_photobleaching_test.'+lista[-1]
                    savemat(name_file,mat)
                    del mat
                # check if deep or traditional features have to be extracted
                if (self.deep_features==1) and (self.traditional_features==0):
                    # extract deep features from the modified dataset    
                    print('Calculating Deep Feature modified by photobleaching test...')
                    feat_mod = model.predict(imds_mod)
                elif (self.deep_features==0) and (self.traditional_features==1):
                    # extract deep features from the modified dataset
                    print('Calculating Traditional Feature modified by photobleaching test...')
                    feat_mod = extract_standard_descriptors(imds_mod)
                # compute the feature selection: here there is the heart of the algorithm (inside utils file)                
                print('Calculating DP and sensitivity for feature selection...')
                feat_sel, SENS, DP_mod, DP_0, dict_for_plot = my_feature_selection(self.feat, feat_mod, GT, th_DP, th_SENS, ind)
                # store the id of the selected features inside the feat_sel_photobleach variable
                self.feat_sel_photobleach = feat_sel
                # store utility variables useful for SENS-DP scatter plots
                self.a_photobleach = dict_for_plot["a"]
                self.b_photobleach = dict_for_plot["b"]
                self.id1_photobleach = dict_for_plot["id1"]
                self.id2_photobleach = dict_for_plot["id2"]
                self.id3_photobleach = dict_for_plot["id3"]
                self.id_photobleach = dict_for_plot["id_"]
                self.DP0_photobleach = dict_for_plot["DP0"]
                self.th_DP_photobleach = dict_for_plot["th_DP"]
                self.th_SENS_photobleach = dict_for_plot["th_SENS"]
            if ind == 2: # test focus
                # perform the focus test and result a set of modified images
                print('Applying 3D phase-contrast fluo saturation test to image dataset...')
                imds_mod = test_saturation(lev_th_fluo, imds, fac)
                # if modified images are intended to be saved save them with their corresponding labels
                if self.var1:
                    mat = {}
                    mat["images_class"] = imds_mod
                    mat["GT"] = GT
                    lista = self.var_modified_dataset_path.split('.')
                    # add the test performed to the name
                    name_file = lista[0]+'_fluo_saturation_test.'+lista[-1]
                    savemat(name_file,mat)
                    del mat
                # check if deep or traditional features have to be extracted
                if (self.deep_features==1) and (self.traditional_features==0):
                    # extract deep features from the modified dataset    
                    print('Calculating Deep Feature modified by fluo saturation test...')
                    feat_mod = model.predict(imds_mod)
                elif (self.deep_features==0) and (self.traditional_features==1):
                    # extract deep features from the modified dataset
                    print('Calculating Traditional Feature modified by fluo saturation test...')
                    feat_mod = extract_standard_descriptors(imds_mod)
                
                # compute the feature selection: here there is the heart of the algorithm (inside utils file) 
                print('Calculating DP and sensitivity for feature selection...')
                feat_sel, SENS, DP_mod, DP_0, dict_for_plot = my_feature_selection(self.feat, feat_mod, GT, th_DP, th_SENS, ind)
                # store the id of the selected features inside the feat_sel_fluo_saturation variable
                self.feat_sel_fluo_saturation = feat_sel
                # store utility variables useful for SENS-DP scatter plots
                self.a_saturation = dict_for_plot["a"]
                self.b_saturation = dict_for_plot["b"]
                self.id1_saturation = dict_for_plot["id1"]
                self.id2_saturation = dict_for_plot["id2"]
                self.id3_saturation = dict_for_plot["id3"]
                self.id_saturation = dict_for_plot["id_"]
                self.DP0_saturation = dict_for_plot["DP0"]
                self.th_DP_saturation = dict_for_plot["th_DP"]
                self.th_SENS_saturation = dict_for_plot["th_SENS"]
        # clear keras session
        tf.keras.backend.clear_session()
    
    
    def phase_contrast_tests(self):
        # instantiate network
        net = select_network(self.var_net_name)
        # instantiate layer
        layer = self.var_layer_name
        # select dimensions
        fac = net.layers[0].input.shape[1:3]
        # take string for lev_bright and transform it in numpy array
        string = self.var_lev_bright
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_bright = np.arange(start,stop, step)
        del string
        # take string for lev_rot and transform it in numpy array
        string = self.var_lev_rot
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_rot = np.arange(start,stop, step)
        del string
        # take string for lev_lambda_pattern and transform it in numpy array
        string = self.var_lev_lambda_pattern
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_lambda_pattern = np.arange(start,stop, step)
        del string
        # take string for lev_focus and transform it in numpy array
        string = self.var_lev_focus
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_focus = np.arange(start,stop, step)
        del string
        # take string for th_DP and th_SENS and transform it in float numbers
        th_DP = float(self.var_th_DP)
        th_SENS = float(self.var_th_SENS)
        # create a transfer learning model with the selected layer adn network
        model = construct_transfer_learning_model(net, layer)
        # select the path to the images
        file_path = self.var_class_path
        # store the first image dataset paths list
        imds = list(paths.list_images(file_path)) #list of the path to the images
        # read all the images and associated labels
        imds_im, GT = read_all(imds, fac) #images as numpy arrays
        # if normal images are intended to be saved save them with their corresponding labels
        if self.var1:
            print('Saving normal image dataset')
            mat = {}
            mat["images_class"] = imds_im
            mat["GT"] = GT
            savemat(self.var_normal_dataset_path,mat)
            del mat
        
        # check if deep or traditional features have to be extracted
        if (self.deep_features==1) and (self.traditional_features==0):
            # extract features from normal images
            print('Calculating Deep Feature dataset original...')
            self.feat = model.predict(imds_im)
        elif (self.deep_features==0) and (self.traditional_features==1):
            # extract features from normal images
            print('Calculating Traditional Feature dataset original...')
            self.feat = extract_standard_descriptors(imds_im)
        # store Ground truth labels in GT attribute
        self.GT = GT
        # loop over tests
        for ind in range(3):
            # label for visualization
            var_process = "Performing test " + str(ind+1) + " of 3"
            self.lbl_1.config(text=var_process)
            if ind==0: #test brightness
                # perform the focus test and result a set of modified images
                print('Applying 3D phase-contrast luminance test to image dataset...')
                imds_mod = test_luminance_lamp(lev_bright, imds, fac)
                # if modified images are intended to be saved save them with their corresponding labels
                if self.var1:
                    mat = {}
                    mat["images_class"] = imds_mod
                    mat["GT"] = GT
                    lista = self.var_modified_dataset_path.split('.')
                    # add the test performed to the name
                    name_file = lista[0]+'_brightness_test.'+lista[-1]
                    savemat(name_file,mat)
                    del mat
                # check if deep or traditional features have to be extracted
                if (self.deep_features==1) and (self.traditional_features==0):
                    # extract deep features from the modified dataset 
                    print('Calculating Deep Feature modified by luminance test...')
                    feat_mod = model.predict(imds_mod)
                elif (self.deep_features==0) and (self.traditional_features==1):
                    # extract deep features from the modified dataset
                    print('Calculating Traditional Feature modified by luminance test...')
                    feat_mod = extract_standard_descriptors(imds_mod)
                # compute the feature selection: here there is the heart of the algorithm (inside utils file) 
                print('Calculating DP and sensitivity for feature selection...')
                feat_sel, SENS, DP_mod, DP_0, dict_for_plot = my_feature_selection(self.feat, feat_mod, GT, th_DP, th_SENS, ind)
                # store the id of the selected features inside the feat_sel_brightness variable
                self.feat_sel_brightness = feat_sel
                # store utility variables useful for SENS-DP scatter plots
                self.a_luminance = dict_for_plot["a"]
                self.b_luminance = dict_for_plot["b"]
                self.id1_luminance = dict_for_plot["id1"]
                self.id2_luminance = dict_for_plot["id2"]
                self.id3_luminance = dict_for_plot["id3"]
                self.id_luminance = dict_for_plot["id_"]
                self.DP0_luminance = dict_for_plot["DP0"]
                self.th_DP_luminance = dict_for_plot["th_DP"]
                self.th_SENS_luminance = dict_for_plot["th_SENS"]
                
            if ind==1: #test movement
                # perform the focus test and result a set of modified images    
                print('Applying 3D phase-contrast gel-pattern test to image dataset...')
                imds_mod = test_gel_pattern(lev_rot,lev_lambda_pattern,imds, fac)
                # if modified images are intended to be saved save them with their corresponding labels
                if self.var1:
                    mat = {}
                    mat["images_class"] = imds_mod
                    mat["GT"] = GT
                    lista = self.var_modified_dataset_path.split('.')
                    # add the test performed to the name
                    name_file = lista[0]+'_gel_pattern_test.'+lista[-1]
                    savemat(name_file,mat)
                    del mat
                # check if deep or traditional features have to be extracted
                if (self.deep_features==1) and (self.traditional_features==0): 
                    # extract deep features from the modified dataset 
                    print('Calculating Deep Feature modified by gel-pattern test...')
                    feat_mod = model.predict(imds_mod)
                elif (self.deep_features==0) and (self.traditional_features==1):
                    # extract deep features from the modified dataset
                    print('Calculating Traditional Feature modified by gel-pattern test...')
                    feat_mod = extract_standard_descriptors(imds_mod)
                # compute the feature selection: here there is the heart of the algorithm (inside utils file) 
                print('Calculating DP and sensitivity for feature selection...')
                feat_sel, SENS, DP_mod, DP_0, dict_for_plot = my_feature_selection(self.feat, feat_mod, GT, th_DP, th_SENS, ind)
                # store the id of the selected features inside the feat_sel_gel_pattern variable
                self.feat_sel_gel_pattern = feat_sel
                # store utility variables useful for SENS-DP scatter plots
                self.a_gel_pattern = dict_for_plot["a"]
                self.b_gel_pattern = dict_for_plot["b"]
                self.id1_gel_pattern = dict_for_plot["id1"]
                self.id2_gel_pattern = dict_for_plot["id2"]
                self.id3_gel_pattern = dict_for_plot["id3"]
                self.id_gel_pattern = dict_for_plot["id_"]
                self.DP0_gel_pattern = dict_for_plot["DP0"]
                self.th_DP_gel_pattern = dict_for_plot["th_DP"]
                self.th_SENS_gel_pattern = dict_for_plot["th_SENS"]
                
            if ind == 2: # test focus
                # perform the focus test and result a set of modified images
                print('Applying 3D phase-contrast out-of-focus test to image dataset...')
                imds_mod = test_out_of_focus(lev_focus, imds, fac)
                # if modified images are intended to be saved save them with their corresponding labels
                if self.var1:
                    mat = {}
                    mat["images_class"] = imds_mod
                    mat["GT"] = GT
                    lista = self.var_modified_dataset_path.split('.')
                    # add the test performed to the name
                    name_file = lista[0]+'_out_of_focus_test.'+lista[-1]
                    savemat(name_file,mat)
                    del mat
                # check if deep or traditional features have to be extracted
                if (self.deep_features==1) and (self.traditional_features==0):
                    # extract deep features from the modified dataset
                    print('Calculating Deep Feature modified by out-of-focus test...')
                    feat_mod = model.predict(imds_mod)
                elif (self.deep_features==0) and (self.traditional_features==1):
                    # extract deep features from the modified dataset
                    print('Calculating Traditional Feature modified by out-of-focus test...')
                    feat_mod = extract_standard_descriptors(imds_mod)
                # compute the feature selection: here there is the heart of the algorithm (inside utils file) 
                print('Calculating DP and sensitivity for feature selection...')
                feat_sel, SENS, DP_mod, DP_0, dict_for_plot = my_feature_selection(self.feat, feat_mod, GT, th_DP, th_SENS, ind)
                # store the id of the selected features inside the feat_sel_out_of_focus variable
                self.feat_sel_out_of_focus = feat_sel
                # store utility variables useful for SENS-DP scatter plots
                self.a_out_of_focus = dict_for_plot["a"]
                self.b_out_of_focus = dict_for_plot["b"]
                self.id1_out_of_focus = dict_for_plot["id1"]
                self.id2_out_of_focus = dict_for_plot["id2"]
                self.id3_out_of_focus = dict_for_plot["id3"]
                self.id_out_of_focus = dict_for_plot["id_"]
                self.DP0_out_of_focus = dict_for_plot["DP0"]
                self.th_DP_out_of_focus = dict_for_plot["th_DP"]
                self.th_SENS_out_of_focus = dict_for_plot["th_SENS"]
        # clear keras session        
        tf.keras.backend.clear_session()
    
    

    def brightfield_tests(self):
        
        # instantiate network
        net = select_network(self.var_net_name)
        #instantiate layer
        layer = self.var_layer_name
        # select dimensions
        fac = net.layers[0].input.shape[1:3]
        # take string for lev_bright and transform it in numpy array
        string = self.var_lev_bright
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_bright = np.arange(start,stop, step)
        del string
        # take string for lev_rot and transform it in numpy array
        string = self.var_lev_rot
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_rot = np.arange(start,stop, step)
        del string
        # take string for lev_trasl and transform it in numpy array
        string = self.var_lev_trasl
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_trasl = np.arange(start,stop, step)
        del string
        # take string for lev_focus and transform it in numpy array
        string = self.var_lev_focus
        string = string.split('[')[-1]
        string = string.split(']')[0]
        lista = string.split(':')        
        start = float(lista[0])
        step = float(lista[1])
        stop = float(lista[2])+step
        lev_focus = np.arange(start,stop, step)
        del string
        # take string for var_th_DP and var_th_SENS and transform it in float numbers
        th_DP = float(self.var_th_DP)
        th_SENS = float(self.var_th_SENS)
        # creating the transfer learning model with the selected layer
        model = construct_transfer_learning_model(net, layer)
        # select the path to the images
        file_path = self.var_class_path
        # store the first image dataset list
        imds = list(paths.list_images(file_path)) #list of the path to the images
        # read all the images and associated labels
        imds_im, GT = read_all(imds, fac) #images as numpy arrays
        # if normal images are intended to be saved save them with their corresponding labels
        if self.var1:
            print('Saving normal image dataset')
            mat = {}
            mat["images_class"] = imds_im
            mat["GT"] = GT
            savemat(self.var_normal_dataset_path,mat)
            del mat
        # check if deep or traditional features have to be extracted
        if (self.deep_features==1) and (self.traditional_features==0):
            # extract features from normal images
            print('Calculating Deep Feature dataset original...')
            self.feat = model.predict(imds_im)
        elif (self.deep_features==0) and (self.traditional_features==1):
            # extract features from normal images
            print('Calculating Traditional Feature dataset original...')
            self.feat = extract_standard_descriptors(imds_im)
        # store Ground truth labels in GT attribute
        self.GT = GT
        # loop over tests        
        for ind in range(3):
            # label for visualization
            var_process = "Performing test " + str(ind+1) + " of 3"
            self.lbl_1.config(text=var_process)
            if ind==0: #test brightness
                # perform the luminance test and result a set of modified images
                print('Applying 2D brightfield luminance test to image dataset...')
                imds_mod = test_luminance_lamp(lev_bright, imds, fac)
                # if modified images are intended to be saved save them with their corresponding labels
                if self.var1:
                    mat = {}
                    mat["images_class"] = imds_mod
                    mat["GT"] = GT
                    lista = self.var_modified_dataset_path.split('.')
                    # add the test performed to the name
                    name_file = lista[0]+'_brightness_test.'+lista[-1]
                    savemat(name_file,mat)
                    del mat
                # check if deep or traditional features have to be extracted
                if (self.deep_features==1) and (self.traditional_features==0):
                    # extract deep features from the modified dataset    
                    print('Calculating Deep Feature modified by luminance test...')
                    feat_mod = model.predict(imds_mod)
                elif (self.deep_features==0) and (self.traditional_features==1):
                    # extract deep features from the modified dataset
                    print('Calculating Traditional Feature modified by luminance test...')
                    feat_mod = extract_standard_descriptors(imds_mod)
                # compute the feature selection: here there is the heart of the algorithm (inside utils file) 
                print('Calculating DP and sensitivity for feature selection...')
                feat_sel, SENS, DP_mod, DP_0, dict_for_plot = my_feature_selection(self.feat, feat_mod, GT, th_DP, th_SENS, ind)
                # store the id of the selected features inside the feat_sel_brightness variable
                self.feat_sel_brightness = feat_sel
                # store utility variables useful for SENS-DP scatter plots
                self.a_luminance = dict_for_plot["a"]
                self.b_luminance = dict_for_plot["b"]
                self.id1_luminance = dict_for_plot["id1"]
                self.id2_luminance = dict_for_plot["id2"]
                self.id3_luminance = dict_for_plot["id3"]
                self.id_luminance = dict_for_plot["id_"]
                self.DP0_luminance = dict_for_plot["DP0"]
                self.th_DP_luminance = dict_for_plot["th_DP"]
                self.th_SENS_luminance = dict_for_plot["th_SENS"]
                
                
            if ind==1: #test movement
                # perform the luminance test and result a set of modified images
                print('Applying 2D brightfield movement test to image dataset...')
                imds_mod = test_movement(lev_rot,lev_trasl,imds, fac)
                # if modified images are intended to be saved save them with their corresponding labels
                if self.var1:
                    mat = {}
                    mat["images_class"] = imds_mod
                    mat["GT"] = GT
                    lista = self.var_modified_dataset_path.split('.')
                    # add the test performed to the name
                    name_file = lista[0]+'_movement_test.'+lista[-1]
                    savemat(name_file,mat)
                    del mat
                # check if deep or traditional features have to be extracted
                if (self.deep_features==1) and (self.traditional_features==0):
                    # extract deep features from the modified dataset    
                    print('Calculating Deep Feature modified by movement test...')
                    feat_mod = model.predict(imds_mod)
                elif (self.deep_features==0) and (self.traditional_features==1):
                    # extract deep features from the modified dataset
                    print('Calculating Traditional Feature modified by movement test...')
                    feat_mod = extract_standard_descriptors(imds_mod)
                # compute the feature selection: here there is the heart of the algorithm (inside utils file) 
                print('Calculating DP and sensitivity for feature selection...')
                feat_sel, SENS, DP_mod, DP_0, dict_for_plot = my_feature_selection(self.feat, feat_mod, GT, th_DP, th_SENS, ind)
                # store the id of the selected features inside the feat_sel_movement variable
                self.feat_sel_movement = feat_sel
                # store utility variables useful for SENS-DP scatter plots
                self.a_movement = dict_for_plot["a"]
                self.b_movement = dict_for_plot["b"]
                self.id1_movement = dict_for_plot["id1"]
                self.id2_movement = dict_for_plot["id2"]
                self.id3_movement = dict_for_plot["id3"]
                self.id_movement = dict_for_plot["id_"]
                self.DP0_movement = dict_for_plot["DP0"]
                self.th_DP_movement = dict_for_plot["th_DP"]
                self.th_SENS_movement = dict_for_plot["th_SENS"]
            if ind == 2: # test focus
                # perform the luminance test and result a set of modified images
                print('Applying 2D brightfield out-of-focus test to image dataset...')
                imds_mod = test_out_of_focus(lev_focus, imds, fac)
                # if modified images are intended to be saved save them with their corresponding labels
                if self.var1:
                    mat = {}
                    mat["images_class"] = imds_mod
                    mat["GT"] = GT
                    lista = self.var_modified_dataset_path.split('.')
                    # add the test performed to the name
                    name_file = lista[0]+'_out_of_focus_test.'+lista[-1]
                    savemat(name_file,mat)
                    del mat
                # check if deep or traditional features have to be extracted
                if (self.deep_features==1) and (self.traditional_features==0):
                    # extract deep features from the modified dataset    
                    print('Calculating Deep Feature modified by out-of-focus test...')
                    feat_mod = model.predict(imds_mod)
                elif (self.deep_features==0) and (self.traditional_features==1):
                    # extract deep features from the modified dataset
                    print('Calculating Traditional Feature modified by movement test...')
                    feat_mod = extract_standard_descriptors(imds_mod)
                # compute the feature selection: here there is the heart of the algorithm (inside utils file) 
                print('Calculating DP and sensitivity for feature selection...')
                feat_sel, SENS, DP_mod, DP_0, dict_for_plot = my_feature_selection(self.feat, feat_mod, GT, th_DP, th_SENS, ind)
                # store the id of the selected features inside the feat_sel_out_of_focus variable
                self.feat_sel_out_of_focus = feat_sel
                # store utility variables useful for SENS-DP scatter plots
                self.a_out_of_focus = dict_for_plot["a"]
                self.b_out_of_focus = dict_for_plot["b"]
                self.id1_out_of_focus = dict_for_plot["id1"]
                self.id2_out_of_focus = dict_for_plot["id2"]
                self.id3_out_of_focus = dict_for_plot["id3"]
                self.id_out_of_focus = dict_for_plot["id_"]
                self.DP0_out_of_focus = dict_for_plot["DP0"]
                self.th_DP_out_of_focus = dict_for_plot["th_DP"]
                self.th_SENS_out_of_focus = dict_for_plot["th_SENS"]
        # clear keras session 
        tf.keras.backend.clear_session()
        
        

    def show_results(self):
        # this method is ilnked to btn8 "choose features"
        
        # open a new window
        self.window2 = Tk()
        # title it Results
        self.window2.title('Results')
        # define a geometry
        self.window2.geometry("500x200+10+20")
        # put this winndow on top
        self.window2.lift(self.window)
        # define the column of the ceck button
        column_check_button = 1
        # insert the label
        self.lbl7 = Label(self.window2,text='Select the features you want to use in further analysis')
        # position the label inside the window
        self.lbl7.grid(column=column_check_button,row=0, sticky='nsew')
        # define the number of rows and columns and configure the window for visualization
        n_cols = 3
        n_rows = 4
        for r  in range(n_rows):
            Grid.rowconfigure(self.window2, r, weight=1)
        for c in range(n_cols):
            Grid.columnconfigure(self.window2, c, weight=1)
        # set the three attributes var2 var3 and var4 to false
        self.var2 = False
        self.var3 = False
        self.var4 = False
        # esablish location of the check buttons
        location = W
        if self.var_name_modality =='2D brightfield':
            # check button for brightness test and place it in row # 1
            self.check_btn2 = Checkbutton(self.window2, text= 'Features selected with brightness test', command=self.select_brightness_test_features,variable=IntVar())
            self.check_btn2.grid(column=column_check_button,row=1, sticky=location)
            # check button for movement test and place it in row # 2
            self.check_btn3 = Checkbutton(self.window2, text= 'Features selected with movement test',command=self.select_movement_test_features, variable= IntVar())
            self.check_btn3.grid(column=column_check_button,row=2, sticky=location)
            # check button for out of focus test and place it in row # 3
            self.check_btn4 = Checkbutton(self.window2, text = 'Features selected with out-of-focus test', command = self.select_out_of_focus_test_features, variable= IntVar())
            self.check_btn4.grid(column=column_check_button,row=3, sticky=location)
        elif self.var_name_modality=='3D phase-contrast':
            # check button for brightness test and place it in row # 1
            self.check_btn2 = Checkbutton(self.window2, text='Features selected with brightness test', command=self.select_brightness_test_features,variable=IntVar())
            self.check_btn2.grid(column=column_check_button,row=1, sticky=location)
            # check button for gel-pattern test and place it in row # 2
            self.check_btn3 = Checkbutton(self.window2, text= 'Features selected with gel-pattern test',command=self.select_gel_pattern_test_features, variable= IntVar())
            self.check_btn3.grid(column=column_check_button,row=2, sticky=location)
            # check button for out of focus test and place it in row # 3
            self.check_btn4 = Checkbutton(self.window2, text = 'Features selected with out-of-focus test', command = self.select_out_of_focus_test_features, variable= IntVar())
            self.check_btn4.grid(column=column_check_button,row=3, sticky=location)
        elif self.var_name_modality=='3D fluorescence':
            # check button for autofluorescence test and place it in row # 1
            self.check_btn2 = Checkbutton(self.window2, text='Features selected with autofluorescence test', command=self.select_autofluorescence_test_features,variable=IntVar())
            self.check_btn2.grid(column=column_check_button,row=1, sticky=location)
            # check button for photobleaching test and place it in row # 2
            self.check_btn3 = Checkbutton(self.window2, text= 'Features selected with photobleaching test',command=self.select_photobleaching_test_features, variable= IntVar())
            self.check_btn3.grid(column=column_check_button,row=2, sticky=location)
            # check button for saturation test and place it in row # 3
            self.check_btn4 = Checkbutton(self.window2, text = 'Features selected with saturation test', command = self.select_saturation_test_features, variable= IntVar())
            self.check_btn4.grid(column=column_check_button,row=3, sticky=location)
        
        # create a Button to actually select the features and place it in column on the right and row 1
        self.btn_7 = Button(self.window2,text='Select features', command=self.select_features)
        self.btn_7.grid(column=column_check_button+1,row=1, sticky=location)
        # put the icon on the window        
        self.window2.iconbitmap(icon_name)
        self.window2.mainloop()
    def select_features(self):
        # this method is linked to to btn_7
        
        # instantiate an empty list
        feat_sel = []
        if self.var_name_modality == '2D brightfield':
            # if one of the attributes are switched to true append it to the list feat_sel
            if self.var2:
                feat_sel.append(self.feat_sel_brightness)
            if self.var3:
                feat_sel.append(self.feat_sel_movement)
            if self.var4:
                feat_sel.append(self.feat_sel_out_of_focus)
        elif self.var_name_modality == '3D phase-contrast':
            # if one of the attributes are switched to true append it to the list feat_sel
            if self.var2:
                feat_sel.append(self.feat_sel_brightness)
            if self.var3:
                feat_sel.append(self.feat_sel_gel_pattern)
            if self.var4:
                feat_sel.append(self.feat_sel_out_of_focus)
        elif self.var_name_modality == '3D fluorescence':
            # if one of the attributes are switched to true append it to the list feat_sel
            if self.var2:
                feat_sel.append(self.feat_sel_autofluorescence)
            if self.var3:
                feat_sel.append(self.feat_sel_photobleach)
            if self.var4:
                feat_sel.append(self.feat_sel_fluo_saturation)
        
        try:
            # if the length is different from zero
            assert len(feat_sel)!=0
            # take the first element from the list and remove it
            features_selected = feat_sel.pop(0)
        except:    
            # show a warning message
            tkinter.messagebox.showwarning(title='Warning',message='please select at least one type of features selected')
            
            return
        # if feat_sel has length = 1
        if len(feat_sel)!=1:
            # loop over the list
            for feat_selected in feat_sel:
                # concatenate the feat_selected to the features_selected
                features_selected = np.concatenate((features_selected,feat_selected),axis =0)
        
        try:
            assert len(features_selected)!=0
            # take features id only once  
            self.features_selected = np.unique(features_selected)
            # instantiate the normal
            self.btn11["state"] = NORMAL
            self.btn12["state"] = NORMAL
            # destroy the window
            self.window2.destroy()
        except: 
            # show a warning message
            tkinter.messagebox.showwarning(title='Warning',message='please select a non empty set of features. If you already selected all the features sets, consider changing the setting parameters')
            self.btn11["state"] = DISABLED
            self.btn12["state"] = DISABLED
            # destroy the window
            self.window2.destroy()
        
    def save_results(self):
        # this method is linked to btn13
        try:
            # if var_DM_parameter_path, var_training_dataset_path, var_test_dataset_path, var_test_path
            assert (self.var_DM_parameter_path!='') and (self.var_training_dataset_path!='') and (self.var_test_dataset_path!='') and (self.var_test_path!='')
        except:
            # if it is not true show a warning
            messagebox.showwarning(title='Warning', message='please select a the test files, path to training and test features and to parameters of the algorithm')
            return
        # instantiate three dictionaries
        mat_train = {}
        mat_test = {}
        mat_parameters = {}
        # select the features for the training set
        feat_train = self.feat[:,self.features_selected]
        # store it inside train dictionary
        mat_train["features"] = feat_train
        # and associated labels
        mat_train["GT"] = self.GT
        # instantiate network and layer 
        net = select_network(self.var_net_name)
        layer = self.var_layer_name
        # take the input shape
        fac = net.layers[0].input.shape[1:3]
        # creating the transfer learning model with the selected layer
        model = construct_transfer_learning_model(net, layer)
        # create a list of path to test images
        imds_test = list(paths.list_images(self.var_test_path))
        # read all test images and associated labels
        imds_test_im, GT_test = read_all(imds_test,fac)
        # check if deep or traditional features have to be extracted
        if (self.deep_features==1) and (self.traditional_features==0):
            # extract the features from the images
            feat_test_all = model.predict(imds_test_im)
        elif (self.deep_features==0) and (self.traditional_features==1):
            # extract features from normal images
            feat_test_all = extract_standard_descriptors(imds_test_im)
            names_features = []
            names_features.append("mean intensity")
            names_features.append("median intensity")
            names_features.append("standard deviation")
            names_features.append("Quantile 10 %")
            names_features.append("Quantile 25 %")
            names_features.append("Quantile 75 %")
            names_features.append("Quantile 90 %")
            names_features.append("Maximum intensity")
            names_features.append("Minimum intensity")
            names_features.append("Shannon Entropy of the Intensity")
            for i in range(60):
                names_features.append("Haralick Texture Feature #"+str(i+1))
            names_features_selected = []
            for i in self.features_selected:
                names_features_selected.append(names_features[i])
            
            mat_parameters["names_features"] = names_features
            mat_parameters["names_features_selected"] = names_features_selected
        
        # select the features for the test set
        feat_test = feat_test_all[:,self.features_selected]
        # store selected features inside test dictionary
        mat_test["features"] = feat_test
        # store labels inside test dictionary
        mat_test["GT"] = GT_test
        
        # save all informations about the algorithm inside the mat_parameters dictionary
        mat_parameters["modality"] = self.var_name_modality
        mat_parameters["path_to_setting_file"] = self.var_setting_file
        mat_parameters["path_to_training_file"] = self.var_class_path
        mat_parameters["deep_features"] = self.deep_features
        mat_parameters["traditional_features"] = self.traditional_features
        # if the images are intended to be saved store the path to normal and modified datasets
        if self.var1:
            mat_parameters["path_to_normal_dataset"] = self.var_normal_dataset_path
            mat_parameters["path_to_modified_dataset"] = self.var_modified_dataset_path
        # store the net name
        mat_parameters["net_name"] = self.var_net_name
        # store the layer name
        mat_parameters["layer_name"] = self.var_layer_name
        # store the th_DP
        mat_parameters["th_DP"] = self.var_th_DP
        # store the th_SENS
        mat_parameters["th_SENS"] = self.var_th_SENS
        # store the path to test images
        mat_parameters["path_to_test_images"] = self.var_test_path
        # store the path to train dataset containing 
        mat_parameters["path_to_train_dataset_with_selected_features"] = self.var_training_dataset_path
        # store the path to test dataset containing 
        mat_parameters["path_to_test_dataset_with_selected_features"] = self.var_test_dataset_path
        if self.var_name_modality == '2D brightfield':
            # store the parameters inside the dictionary mat_parameters
            mat_parameters["lev_luminance"] = self.var_lev_bright
            mat_parameters["lev_rotation"] = self.var_lev_rot
            mat_parameters["lev_focus"] = self.var_lev_focus
            mat_parameters["lev_trasl"] = self.var_lev_trasl
            mat_parameters["feat_sel_brightness"] = self.feat_sel_brightness
            mat_parameters["feat_sel_movement"] = self.feat_sel_movement
            mat_parameters["feat_sel_out_of_focus"] = self.feat_sel_out_of_focus
            mat_parameters["feat_brightness_considered"] = self.var2
            mat_parameters["feat_movement_considered"] = self.var3
            mat_parameters["feat_out_of_focus_considered"] = self.var4
            mat_parameters["feat_sel_union"] = self.features_selected
        if self.var_name_modality == '3D phase-contrast':
            # store the parameters inside the dictionary mat_parameters
            mat_parameters["lev_luminance"] = self.var_lev_bright
            mat_parameters["lev_rotation"] = self.var_lev_rot
            mat_parameters["lev_focus"] = self.var_lev_focus
            mat_parameters["lev_lambda_pattern"] = self.var_lev_lambda_pattern
            mat_parameters["feat_sel_brightness"] = self.feat_sel_brightness
            mat_parameters["feat_sel_gel_pattern"] = self.feat_sel_gel_pattern
            mat_parameters["feat_sel_out_of_focus"] = self.feat_sel_out_of_focus
            mat_parameters["feat_brightness_considered"] = self.var2
            mat_parameters["feat_gel_pattern_considered"] = self.var3
            mat_parameters["feat_out_of_focus_considered"] = self.var4
            mat_parameters["feat_sel_union"] = self.features_selected
        if self.var_name_modality == '3D fluorescence':
            # store the parameters inside the dictionary mat_parameters
            mat_parameters["lev_luminance"] = self.var_lev_bright
            mat_parameters["lev_photobleach"] = self.var_lev_perc_bleach 
            mat_parameters["lev_th_fluo"] = self.var_lev_th_fluo
            mat_parameters["feat_sel_autofluorescence"] = self.feat_sel_autofluorescence
            mat_parameters["feat_sel_photobleach"] = self.feat_sel_photobleach
            mat_parameters["feat_sel_fluo_saturation"] = self.feat_sel_fluo_saturation
            mat_parameters["feat_autofluorescence_considered"] = self.var2
            mat_parameters["feat_photobleach_considered"] = self.var3
            mat_parameters["feat_th_fluo_considered"] = self.var4
            mat_parameters["feat_sel_union"] = self.features_selected
            
        # save the three dictionaries 
        savemat(self.var_training_dataset_path, mat_train)
        savemat(self.var_test_dataset_path, mat_test)
        savemat(self.var_DM_parameter_path, mat_parameters)
        # show a message box
        messagebox.showinfo(title='Finish!', message='The process was succesfully completed')
        
    def test_images(self):
        # this method is linked to btn11 
        
        # start getting the current working folder
        current_dir = os.getcwd()
        # ask directory        
        f = filedialog.askdirectory(title='Select the folder containing subdirectories with test images of the two classes',initialdir=current_dir)
        try:
            # if f is  not empty
            assert f!=''
            # select the var_test_path variable
            self.var_test_path = f
        except:
            # show a message box with a warning
            messagebox.showwarning(title='Warning', message='please select a folder')
            return
        
        # split the variable for visualization
        var_test_path2 = f.split('/')
        # select only the last name of the directory
        var_test_path2 = var_test_path2[-1]
        # config the label with the name tested
        self.lbl11.config(text=var_test_path2)
        # set the ceck1 to true
        self.check1 = True
        if self.var1:
            if (self.check1) and (self.check2) and (self.var_normal_dataset_path!='') and (self.var_modified_dataset_path!=''):
                 self.btn13["state"] = NORMAL
            else:
                self.btn13["state"] = DISABLED
        else:
            if (self.check1)and(self.check2):
                self.btn13["state"] = NORMAL
            else:
                self.btn13["state"] = DISABLED
        
    
    def select_brightness_test_features(self):
        # change the state attribute var2
        self.var2 = not(self.var2)
    def select_movement_test_features(self):
        # change the state attribute var3
        self.var3 = not(self.var3)
    def select_out_of_focus_test_features(self):
        # change the state attribute var4
        self.var4 = not(self.var4)
    def select_gel_pattern_test_features(self):
        # change the state attribute var3
        self.var3 = not(self.var3)
    def select_autofluorescence_test_features(self):
        # change the state attribute var2
        self.var2 = not(self.var2)
    def select_photobleaching_test_features(self):
        # change the state attribute var3
        self.var3 = not(self.var3)
    def select_saturation_test_features(self):
        # change the state attribute var4
        self.var4 = not(self.var4)
# instantiate the an element of the GUI class
GUI = GUI()
# call the set_gui method
GUI.set_gui()
