# import necessary packages
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np

from matplotlib import pyplot as plt
import PySimpleGUI as sg
import imutils
import math
import os
import pandas as pd
from math import log, e
from skimage.feature import hog, graycomatrix, graycoprops

def select_network(name):
    # Input Arguments:
    # - name: string containing the name of the desired network for feature extraction
    # Output Arguments:
    # - net: Tensorflow pretrained model of the selected network
    if name == 'resnet101':
        net = keras.applications.ResNet101(include_top=True,weights='imagenet')
        # Video Print infos about the net selected
        print('[INFO]: the selected net is ResNet101')
        with open('Infos.txt','w+') as f:
            f.write('[INFO]: the selected net is ResNet101')
            f.close()        
    elif name == 'vgg19':
        net = keras.applications.VGG19(include_top=True,weights='imagenet')
        print('[INFO]: the selected net is VGG19')
        with open('Infos.txt','w') as f:
            f.write('[INFO]: the selected net is VGG19')
            f.close()
    elif name == 'nasnetlarge':
        net = keras.applications.NASNetLarge(include_top=True,weights='imagenet')
        print('[INFO]: the selected net is NasNETLarge')
        with open('Infos.txt','w+') as f:
            f.write('[INFO]: the selected net is NasNETLarge')
            f.close()
    elif name == 'densenet201':
        net = keras.applications.DenseNet201(include_top=True,weights='imagenet',input_shape=(224,224,3))
        print('[INFO]: the selected net is DenseNet201')
        with open('Infos.txt','w+') as f:
            f.write('[INFO]: the selected net is DenseNet201')
            f.close()
    return net


# to inspect the layers in the network
#     for layer in net.layers:
#       print(layer.name)
##plot_model(net,'resnet_101.png',show_layer_names=True)

def construct_transfer_learning_model(net, layer):
    # Input Arguments
    # - net: net containing the desired layer and the graph
    # - layer: name of the desired output layer
    # Output Arguments
    # - model: Tensorflow model with image as input and
    #           features of the desired layer as output

    # take the tensor corresponding to the input
    input_image = net.layers[0].input
    # Display infos about the input dimension
    print('[INFO]: layer of the selected network is '+str(layer))
    print('[INFO]: input shape to the selected network is '+str(input_image.shape))
    with open('Infos.txt','a+') as f:
            f.write('\n')
            f.write('[INFO]: input shape to the selected network is '+str(input_image.shape))
            f.write('\n')
            f.write('[INFO]: layer of the selected network is '+str(layer))  
            f.close()
    # take the tensor corresponding to the desired layer for transfer learning
    output_features = net.get_layer(name=layer).output
    # check if the dimension of the output is already flattened
    if not(output_features.shape==2):
        # if it is not flattened than flatten it
        # compute the product of the dimensions of the array except the first dimension (None)
        shape = tf.math.reduce_prod(output_features.shape[1:]).numpy()
        # reshape the output_features 
        output_features = tf.reshape(output_features,[-1,shape])
    
    print('[INFO]: ouput shape of the extracted features '+str(output_features.shape))
    with open('Infos.txt','a+') as f:
            f.write('\n')
            f.write('[INFO]: ouput shape of the extracted features '+str(output_features.shape))
            f.close()
    # construct a model that has input and output corresponding to the two tensors
    # according to the existing graph.
    model = Model(inputs = input_image, outputs = output_features)
    return model

def extract_selected_features(data_in, selected_features, model):
    # Input Arguments
    # - data_in: numpy array containing input images. data_in has shape (#samples, height, width, channels)
    # - selected_features: numpy vector containing the indexes pointing to the selected features.
    # - model: keras model used in the feature selection phase
    # Output Arguments
    # - data_out: numpy array containing the selected features extracted from the input images
    #             data_out has shape (#samples, len(selected_features))
    
    # first extract all features from data_in using the given model    
    feat_extracted = model.predict(data_in)
    
    data_out = feat_extracted[:,selected_features]
    
    return data_out
def my_read_resize(input_path, fac):
    # Input Arguments:
    # - input_path: path to the input image to load from disk and resize
    # - fac: desired output dimension of the array
    # Output Arguments:
    # - image2: array with dimensions (fac[0], fac[1], 3)

    # load the image from disk
    image = cv2.imread(input_path)
    # if the data type is uint8 as it is supposed to be
    # convert image in float 32 because opencv supports only float 32bit
    # (64bit corresponds to matlab double)
    # and rescale between 0 and 1, devinding by 255
    if image.dtype == 'uint8':
        image = image.astype('float32')/255.0
    # if the image has three channels convert to only one channel
    if len(image.shape) == 3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # if the image has not the desired shape (required by the network)
    # reshape it.
    if not(fac[0] == image.shape[0]) or not(fac[1] == image.shape[1]):
        image = cv2.resize(image,(fac[0],fac[1]), interpolation=cv2.INTER_CUBIC)
    # now image is a numpy array of shape (fac[0], fac[1])
    # add a dimension to concatenate matrices:
    # image.shape() = (fac[0], fac[1], 1)
    image = np.expand_dims(image, axis=-1)
    # concatenate the same array across the last  (third) dimension.
    # the result is a fake rbg image
    image = np.concatenate((image,image,image),axis=-1)
    # rescale again in [0, 255] and transform the data type in uint8
    image2 = (image*255.0).astype('uint8')
    # return the array containing the image processed
    return image2
def read_all(input_paths, fac):
    # Input Arguments:
    # - input_paths: list containing the paths to the images to read
    # - fac: desired dimensions (selected based on the input dimensions of the net)
    # Output Arguments:
    # - imds1: Output array of dimensions (#of_images, fac[0], fac[1], 3)
    
    # initialize empty list for the images
    imds1 = []
    # loop over the provided paths
    for i, input_path in enumerate(input_paths):
        # Instantiate and update the waitbar for visualization
        sg.one_line_progress_meter('Load Image Progress Meter...', i+1, len(input_paths), '-key-')
        # read and resize the image corresponding to the provided path
        im = my_read_resize(input_path, fac)
        # append the image to the list
        imds1.append(im)
        del im
    # convert the list in a numpy array
    imds1 = np.array(imds1)
    
    # get the label vector
    labels = [p.split(os.path.sep)[-2] for p in input_paths]
    # instantiate a member of label encoder class
    le = LabelEncoder()
    # create the vector containing the two class
    labels = le.fit_transform(labels)
    # return the array and the classes
    return imds1, labels
def my_feature_selection(feat, feat_mod, GT, th_DP, th_SENS, test_ind, vis=False):
    # Input Arguments:
    # - feat1: features extracted from the non modified images of class1. array (#measures, dimension_features).
    # - feat1_mod: features extracted from the modified images of class1. array (#measures, dimension_features).
    # - feat2: features extracted from the non modified images of class2. array (#measures, dimension_features).
    # - feat1_mod2: features extracted from the modified images of class2. array (#measures, dimension_features).
    # - th_DP: float number. minimum Discriminative Power DP of the features both in
    #           non modified and modified datasets.
    # - th_SENS: float number. maximum sensitivity of the features to the modification test.
    # - test_ind: progressive id corresponding to the test. possible values: 0, 1, 2.
    # Optional Argument:
    # - vis: If to show plots. default=True.

    
    # initialize a zero-vector for keeping AUC values of both normal and modified images
    AUC = np.zeros((feat.shape[1],1))
    AUC_mod = np.zeros((feat.shape[1],1))
    # first compute the AUC for the two set of features
    print("Calculating AUC values...")
    # loop over the features and for each compute auc score.
    for f in range(feat.shape[1]):
        # compute auc for the fth features of normal images
        auc = roc_auc_score(GT,feat[:,f])
        # store the auc value inside the AUC array
        AUC[f] = auc
        # delete the auc value to avoid overwriting errors or bugs
        del auc
    
    print("Calculating AUC values for the modified dataset...")
    for f in range(feat_mod.shape[1]):
        # compute auc for the fth features of modified images
        auc = roc_auc_score(GT,feat_mod[:,f])
        AUC_mod[f] = auc
        del auc
    # stack the two arrays AUC and 1-AUC along the 2nd dimension
    AUC_glob = np.concatenate((AUC,1-AUC),axis=1)
    AUC_glob_mod = np.concatenate((AUC_mod, 1-AUC_mod),axis=1)
    # compute the discriminative power of the features for both normal and modified images
    DP0 = np.max(AUC_glob,axis=1)
    DP_mod = np.max(AUC_glob_mod, axis=1)
    # compute the sensitivity of the features to the applied modification
    # as the absolute difference divided by the initial value. 
    SENS = np.abs(DP_mod-DP0)/DP0
    # select features with Discriminative Power above the selected th_DP threshold
    # for the set of unmodified images
    id_ = np.where(DP0>th_DP)[0]
    
    
    # select the new DP after test application
    a = DP_mod[id_]
    # select the relative variation associated (Sensitivity)
    b = SENS[id_]
    # find id of final selected features (id1) discarded features
    # for sensitivity (id2) or for insufficient DP after modification (id3)
    id1 = logical_operation_between_vectors(a>= th_DP, b<=th_SENS, 'and')
    id2 = logical_operation_between_vectors(a>= th_DP, b> th_SENS, 'and')
    id3 = a< th_DP

    # store the final selected features
    feat_sel = id_[id1]
    
    dict_for_plot = {}
    dict_for_plot["b"] = b
    dict_for_plot["a"] = a
    dict_for_plot["id1"] = id1
    dict_for_plot["id2"] = id2
    dict_for_plot["id3"] = id3
    dict_for_plot["id_"] = id_
    dict_for_plot["DP0"] = DP0
    dict_for_plot["th_DP"] = th_DP
    dict_for_plot["th_SENS"] = th_SENS
    
    
        
    # return the id of the selected features the Sensitivity values and
    # the Discriminative Power of the modified images' features
    return feat_sel, SENS, DP_mod, DP0, dict_for_plot
def get_disk_kernel(r):
    # Input Arguments:
    # r : radius of the circle
    # Output Arguments:
    # ker: array containing the kernel.
    
    # initialize the dimensions of the filter
    kx = int(2*r + 1)
    # we use square filters
    ky = kx
    # inizialize the matrix of the kernel with all zeros
    ker = np.zeros((kx,ky),dtype='float64')
    # find the center of the matrix
    # if r = 10 the center is the point cx = 10, cy =10 (Python is zero indexed)
    cx = kx//2  
    cy = ky//2
    # initialize the counter of true pixels to zero
    n = 0
    # loop over rows
    for row in range(ker.shape[0]):
        # compute the row distance (x distance) from the center
        dx = row - cx
        # loop over the columns
        for col in range(ker.shape[1]):
            # compute the columns distance (y distance) from the center
            dy = col - cy
            # compute the euclidean (L2) distance from the center
            d = math.pow((math.pow(dx,2)+math.pow(dy,2)),0.5) #to check try dx = 3, dy = 4 ==> d = 5
            # if the distance is below the selected radius
            # set the pixel to 1 and update the counter n
            if d<=r:
                ker[row, col] = 1;
                n +=1
            # delete d and dy to avoid overwriting errors or bugs
            del d, dy
        # delete dx to avoid overwriting errors or bugs
        del dx
    # finally divide the obtained matrix by the number of True pixels counted
    ker = ker/n
    # return the constructed array
    return ker
    
def logical_operation_between_vectors(a, b, op):
    # Input arguments:
    # a : first input vector
    # b : second input vector
    # op: string coding the logical operation to be performed. allowed values are 'and', 'or'

    # Output argument:
    # y : output of the logical operation op performed between the elements of
    # vector a and b in order

    # initialize empty list
    y = []
    # loop over the elements of the two arrays
    for (a_, b_) in zip(a,b):
        # append the result of the desired information to the list
        if op=="and":
            y.append(a_ and b_)
        elif op=="or":
            y.append(a_ or b_)
    # convert the list in a numpy array
    y = np.array(y, dtype=bool)
    # return the array
    return y
def test_luminance_lamp(lev_bright, imds1, fac, vis=True):
    # Input Arguments
    # - lev_bright: numpy array containing the available levels of shift in luminance.
    # - imds1: list of paths to the images.
    # - fac: desired input shape ( linked to the input dimensions of the net)
    # Output Arguments:
    # - imds_mod: array containing the modified images. shape = (#images, fac[0], fac[1],3)

    
    # initialize empty list for the arrays
    imds_mod = []
    # loop over the provided paths
    for i, path in enumerate(imds1):
        if vis == True:
            # Instantiate and update the waitbar for visualization
            sg.one_line_progress_meter('Luminance Test Progress Meter', i+1, len(imds1), '-key-')
        
        # read and resize the image corresponding to the provided path
        image = my_read_resize(path, fac)
        # sample a random number for the luminance shift
        id_bright = np.random.randint(low=0, high=len(lev_bright))
        # convert image to float 64 dtype and rescale between 0 and 1
        im = image.astype('float64')/255.0
        # apply shifting in luminance
        im2 = im + lev_bright[id_bright]
        # set to the limits values out of [0, 1]
        im2[im2>1] = 1.0
        im2[im2<0] = 0.0
        # rescale again in [0, 255] and transform the data type in uint8
        im3 = (im2 * 255.0).astype('uint8')
        # append the processed image to the list
        imds_mod.append(im3)
        # delete variables to avoid overwriting errors or bugs
        del im2, im, id_bright, image, im3
    # convert the list in a numpy array
    imds_mod = np.array(imds_mod, dtype='uint8')
    # return the array
    return imds_mod

def test_out_of_focus(lev_focus, imds1, fac,vis=True):
    # Input Arguments:
    # - lev_focus: numpy array containing the available level of shift in focus
    # - imds1: list of paths to the images.
    # - fac: desired input shape ( linked to the input dimensions of the net)
    # Output Arguments:
    # - imds_mod: array containing the modified images. shape = (#images, fac[0], fac[1],3)


    # initialize empty list for the arrays
    imds_mod = []
    # loop over the provided paths
    for i, path in enumerate(imds1):
        if vis == True:
            # Instantiate and update the waitbar for visualization
            sg.one_line_progress_meter('Focus Test Progress Meter', i+1, len(imds1), '-key-')
        
        # read and resize the image corresponding to the provided path
        image = my_read_resize(path, fac)
        # sample a random number for the focus change
        id_focus = np.random.randint(low=0, high=len(lev_focus))
        # convert image to float 64 dtype and rescale between 0 and 1
        im = image.astype('float64')/255.0
        # store the value of the radius in the variable r        
        r = lev_focus[id_focus]
        
        # create kernel
        ker = get_disk_kernel(r)
        # apply the convolution
        im2 = cv2.filter2D(im,-1,ker)
        # rescale again in [0, 255] and transform the data type in uint8
        im2 = (im2*255.0).astype('uint8')
        # append the processed image to the list
        imds_mod.append(im2)
        # delete variables to avoid overwriting errors or bugs
        del im2, im, id_focus, image
    # convert the list in a numpy array
    imds_mod = np.array(imds_mod)
    # return the array
    return imds_mod

def test_movement(lev_rot,lev_trasl,imds1, fac, vis=True):
    # Input Arguments:
    # lev_rot: numpy array containing the available levels of rotation. dtype must be int
    # lev_trasl: numpy array containing the available levels of traslation.
    # imds1: list of paths to the images.
    # - fac: desired input shape ( linked to the input dimensions of the net)
    # Output Arguments:
    # - imds_mod: array containing the modified images. shape = (#images, fac[0], fac[1],3)

    
    # initialize empty list for modified arrays
    imds_mod = []
    # loop over the provided paths
    for i, path in enumerate(imds1):
        if vis == True:
            # Instantiate and update the waitbar for visualization
            sg.one_line_progress_meter('Rototraslation Test Progress Meter', i+1, len(imds1), '-key-')
        
        # read and resize the image corresponding to the provided path
        image = my_read_resize(path, fac)
        # sample a random number for the rotation, traslation and direction
        id_roto = np.random.randint(low=0, high= len(lev_rot))
        id_trasl = np.random.randint(low=0, high= len(lev_trasl))
        id_dim = np.random.randint(low=0, high=2)
        # convert image to float 64 dtype and rescale between 0 and 1
        im = image.astype('float64')/255.0
        # rotate the image according on the sorted number
        im2 = imutils.rotate(im, lev_rot[id_roto])
        # translate the image according to the entity and direction sorted
        if id_dim==0:
            im2 = np.roll(im2, int(lev_trasl[id_trasl]), axis=int(0))
        else:
            im2 = np.roll(im2, int(lev_trasl[id_trasl]), axis=int(1))
        # rescale again in [0, 255] and transform the data type in uint8        
        im2 = (im2*255.0).astype('uint8')
        # append the processed image to the list
        imds_mod.append(im2)
        # delete variables to avoid overwriting errors or bugs
        del im2, im, id_dim, id_trasl, id_roto, image
    # convert the list in a numpy array
    imds_mod = np.array(imds_mod)
    # return the array
    return imds_mod
def make_mask_gel(fac):
    # Input Arguments
    # - fac: iterable with len(fac)=2 containing the desired input shape for the net
    # Output arguments
    # - mask_gel: numpy array containing at each row a sinusoidal signal
    #               with 10 periods.
    
    # construct a sinusoidal signal of 10 periods
    signal = 0.5*(1+np.sin(10*2*np.pi*(np.arange(0,fac[1]))/fac[1]))
    # reshape it into a row vector ==> signal.shape = (1,fac[1])
    signal = np.reshape(signal,(1,signal.shape[0]))
    # insert the first signal as a row in the numpy array mask_gel
    mask_gel = signal
    # loop over all rows of the desired shape (except the one already counted
    for i in range(fac[0]-1):
        # copy the same signal in every row
        mask_gel = np.concatenate((mask_gel,signal),axis=0)
    # add a dimension and concatenate along the last dimension
    mask_gel = np.reshape(mask_gel, (mask_gel.shape[0], mask_gel.shape[1], 1))
    mask_gel = np.concatenate((mask_gel, mask_gel, mask_gel), axis=-1)
    # return the constructed array
    return mask_gel
        

def test_gel_pattern(lev_rot, lev_lambda, imds1, fac,vis=True):
    # Input Arguments:
    # - lev_rot: numpy array containing the possible values of rotation.
    #           must be dtype =int
    # - lev_lambda: numpy array containing the possible values of lambda.
    # - imds1: list of paths to the images.
    # - fac: desired input shape ( linked to the input dimensions of the net).
    # Output Arguments
    # - imds_mod: array containing the modified images. shape = (#images, fac[0], fac[1],3)

    # initialize list for the output images
    imds_mod = []

    # loop over the provided paths
    for i, path in enumerate(imds1):
        if vis == True:
            # Instatiate an update bar for visualization
            sg.one_line_progress_meter("Gel effect progress meter",i+1, len(imds1), '-key')
        
        # read and resize image corresponding to the path
        im = my_read_resize(path,fac)
        # sample a random number for rotation and lambda
        id_roto = np.random.randint(low = 0, high = len(lev_rot))
        id_lambda = np.random.randint(low = 0, high = len(lev_lambda))
        # generate an array with rows as sinusoidal signals
        mask_gel = make_mask_gel(fac)
        # rotate the given array
        mask_gel2 = imutils.rotate(mask_gel, lev_rot[id_roto])
        # multiply by the lambda level
        mask_gel3 = mask_gel2*lev_lambda[id_lambda]
        # convert image in float 64 and scale in [0,1] range
        image = im.astype('float64')/255.0
        # ad the constructed mask
        im2 = image + mask_gel3
        im2[im2>1] = 1.0
        im2[im2<0] = 0.0
        # scale image in range [0, 255] and convert in uint8 data type
        im3 = (im2*255.0).astype('uint8')
        # append the image to the list
        imds_mod.append(im3)
        # delete variables to avoid overwriting errors or bugs
        del im3, im2, image, mask_gel3, mask_gel2, mask_gel, id_lambda, id_roto, im
    # convert the list in numpy array
    imds_mod = np.array(imds_mod)
    # return the constructed array
    return imds_mod

def test_autofluo_luminance_lamp(lev_bright, imds1, fac,vis=True):
    # Input arguments
    # - lev_bright: numpy array containing the desired levels of brightness shift
    # - imds1: list of paths to the input images
    # - fac: desired input shape ( linked to the input dimensions of the net).
    # Output arguments
    # - imds_mod: numpy array containing the modified images. imds_mod.shape= (#images, fac[0], fac[1], 3)

    # initialize list for the output images
    imds_mod = []

    # loop over the provided paths
    for i, path in enumerate(imds1):
        if vis == True:
            # Instatiate an update bar for visualization
            sg.one_line_progress_meter("Autofluorescence luminance lamp progress meter",i+1, len(imds1), '-key')
        
        # read and resize image corresponding to the path
        im = my_read_resize(path,fac)
        # convert the data type to float 32 and scale to range [0, 1]
        im = im.astype('float32')/255.0
        # convert image from rgb (in reality bgr since we used cv2.imread) to gray
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # riconvert in [0, 255] and uint8 in order to perform thresholding
        im2 = (im*255.0).astype('uint8')
        # Otsu Thresholding
        th, im_th = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # again convert in [0,1]
        im_th = im_th.astype('float32')/255.0
        # sample random number for brightness
        id_bright = np.random.randint(low = 0, high = len(lev_bright))
        # compute the leverage average
        lev_avg = lev_bright[id_bright]
        # complement the image and multiply by the leverage average
        bw1 = (1-im_th)*lev_avg
        # sum the obtained image with the original image
        im3 = im + bw1
        im3[im3>1] = 1
        im3[im3<0] = 0
        # apply blurring with gaussian filter ker.shape = (11, 11), sigma = 2
        im3 = cv2.GaussianBlur(im3, (11,11), 2, borderType=cv2.BORDER_REPLICATE)
        # add a dimension to the array and concatenate along the third dimension
        im3 = np.reshape(im3,(im3.shape[0],im3.shape[1],1))
        im3 = np.concatenate((im3,im3,im3), axis=-1)
        # scale in [0,255] and convert to uint8
        im3 = (im3*255.0).astype('uint8')
        # append the image to the list
        imds_mod.append(im3)
        # delete variables to avoid overwriting errors or bugs
        del im3, bw1, lev_avg, id_bright, im_th, th, im2, im
    # convert the list in numpy array
    imds_mod = np.array(imds_mod)
    # return the constructed array
    return imds_mod
def test_photobleaching(perc_bleach, imds1, fac, vis=True):
    # Input Arguments:
    # - perc_bleach: numpy array containing the selected value for the bleaching
    # - imds1: list containing the paths to the images
    # - fac: desired input shape ( linked to the input dimensions of the net).
    # Output Arguments:
    # - imds_mod: numpy array containing the modified images. imds_mod.shape = (#images, fac[0], fac[1], 3)

    # initialize list for the output images
    imds_mod = []

    for i, path in enumerate(imds1):
        if vis == True:
            # Instatiate an update bar for visualization
            sg.one_line_progress_meter("Photobleaching progress meter",i+1, len(imds1), '-key')
        
        # read and resize image corresponding to the path
        im0 = my_read_resize(path,fac)
        
        # convert image from rgb (in reality bgr since we used cv2.imread) to gray
        im0_g = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        # convert the data type to float 32 and scale to range [0, 1]
        im0_f = im0_g.astype('float64')/255.0
        
        # Otsu Thresholding to find the cell
        th, im_th = cv2.threshold(im0_g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # convert the thresholded image in float and [0, 1] range
        im_th_f = im_th.astype('float64')/255.0
        # sample random number for photobleaching
        id_bleach = np.random.randint(low=0, high=len(perc_bleach))
        # multiply elementwise the luminance of the cell by perc_bleach[id_bleach]
        im1 = np.multiply(im0_f,im_th_f*perc_bleach[id_bleach])
        im1[im1>1] = 1
        im1[im1<0] = 0
        # scale in range [0, 255] and data type uint8
        im2 = (im1*255.0).astype('uint8')
        # reshape and concatenate along the 3rd dimension
        im2 = np.reshape(im2,(im2.shape[0],im2.shape[1],1))
        im2 = np.concatenate((im2,im2,im2),axis=-1)
        # append the image to the list
        imds_mod.append(im2)
    #  convert the list in numpy array
    imds_mod = np.array(imds_mod)
    # return the constructed array
    return imds_mod

def test_saturation(th_fluo, imds1, fac,vis=True):
    # Input Arguments
    # - th_fluo: numpy array containing the levels of thresholds
    # - imds1: list containing paths to the images
    # - fac: desired input shape ( linked to the input dimensions of the net).
    # Output Arguments
    # - imds_mod: numpy array containing the modified images. imds_mod.shape = (#images, fac[0], fac[1], 3)

    # initialize list for the output images
    imds_mod = []

    for i, path in enumerate(imds1):
        if vis == True:
            # instantiate an update bar for visualization
            sg.one_line_progress_meter("Saturation progress meter", i+1, len(imds1),'-key-')
        
        # read and resize image corresponding to the path
        im0 = my_read_resize(path,fac)
        # convert image to grayscale
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        # convert the data type to float 32 and scale to range [0, 1]
        im0_f = im0.astype('float64')/255.0
        # Otsu Thresholding to find the cell
        th, im_th = cv2.threshold(im0,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # convert the thresholded image in float and [0, 1] range
        im_th_f = im_th.astype('float64')/255.0
        # multiply elementwise the original float image with the otsu float output
        im1 = np.multiply(im0_f, im_th_f)
        # sample random number for threshold
        id_sat = np.random.randint(low=0, high=len(th_fluo))
        # saturate values over th_fluo to 1
        im1[im1>th_fluo[id_sat]] = 1.0
        im1[im1<0] = 0.0
        # scale to [0, 255] and convert in uint8
        im2 = (im1*255).astype('uint8')
        # reshape and concatenate along the third dimension
        im2 = np.reshape(im2,(im2.shape[0], im2.shape[1],1))
        im2 = np.concatenate((im2,im2,im2),axis=-1)
        # append image th the list
        imds_mod.append(im2)
    # convert the list in numpy array
    imds_mod = np.array(imds_mod)
    # return the constructed array
    return imds_mod
        
def extract_standard_descriptors(imds, vis=True):
    # Input arguments
    # imds: array containig input images
    # Output arguments
    # feat: matrix containing all the features. feat.shape = [#samples, #features=70]
    
    # initialize empty list that will host the features
    feat = []
    
    # loop over the images
    for i, img in enumerate(imds):
        if vis == True:
            # instantiate an update bar for visualization
            sg.one_line_progress_meter("Standard biological features", i+1, len(imds),'-key-')
        
        
        # convert image to grayscale
        im0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert the data type to float 32 and scale to range [0, 1]
        im0_f = im0.astype('float64')/255.0
        # compute the mean of the image and save it in a 1d array
        mean_img = np.array(np.mean(im0_f))
        mean_img = np.reshape(mean_img,(1,))
        # compute the median
        median_img = np.array(np.median(im0_f))
        median_img = np.reshape(median_img,(1,))
        # compute the standard deviation        
        std_img = np.array(np.std(im0_f))
        std_img = np.reshape(std_img,(1,))
        # compute the quantile 0.1        
        q0_1 = np.array(np.quantile(im0_f,0.1))
        q0_1 = np.reshape(q0_1,(1,))
        # compute the quantile 0.25
        q0_25 = np.array(np.quantile(im0_f,0.25))
        q0_25 = np.reshape(q0_25,(1,))
        # compute the quantile 0.75
        q0_75 = np.array(np.quantile(im0_f,0.75))
        q0_75 = np.reshape(q0_75,(1,))
        # compute the quantile 0.9        
        q0_9 = np.array(np.quantile(im0_f,0.9))
        q0_9 = np.reshape(q0_9,(1,))
        # compute the maximum
        max_img_ = np.array(np.max(im0_f))
        max_img_ = np.reshape(max_img_,(1,))
        # compute the minimum
        min_img_ = np.array(np.min(im0_f))
        min_img_ = np.reshape(min_img_,(1,))
        # compute the shannon entropy of the image        
        entropy_img = np.array(entropy(im0))
        entropy_img = np.reshape(entropy_img,(1,))
        # compute the haralick texture features of the image
        contrast, energy, homogeneity, correlation, dissimilarity = haralick_features(im0)
        # concatenate the features along the first axis
        texture_img = np.concatenate((contrast, energy,homogeneity, correlation, dissimilarity), axis = 0)
        # concatenate all the features along the first axis        
        features_img = np.concatenate((mean_img, median_img, std_img, q0_1, q0_25, q0_75, q0_9, max_img_, min_img_, entropy_img, texture_img), axis = 0)
        # append features to the list
        feat.append(features_img)
    # convert the list in numpy array
    feat = np.array(feat)
        
    return feat
    
def haralick_features(gray_image, distances = [1,2,3], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]):
    # Input arguments
    # - gray_image: numpy array containg the gray image on which compute the descriptors 
    # - distances: list of distances used to compute the co occurence matrices
    # - angles: list of angles used to compute the co occurrence matrices
    # Output arguments
    # contrast, energy, homogeneity, correlation, dissimilarity: the computed features
    
    # compute the gray level co occurrence matrix
    glcm = graycomatrix(gray_image, 
                    distances=distances, 
                    angles=angles,
                    symmetric=True,
                    normed=True)
    # properties to be computed 
    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity']
    # compute the statistics proposed by haralick
    contrast = graycoprops(glcm, properties[0])
    energy = graycoprops(glcm, properties[1])
    homogeneity = graycoprops(glcm, properties[2])
    correlation = graycoprops(glcm, properties[3])
    dissimilarity = graycoprops(glcm, properties[4])
    
    # reshape to flattened array every statistic
    contrast = np.reshape(contrast,(np.prod(contrast.shape),))
    energy = np.reshape(energy,(np.prod(energy.shape),))
    homogeneity = np.reshape(homogeneity,(np.prod(homogeneity.shape),))
    correlation = np.reshape(correlation,(np.prod(correlation.shape),))
    dissimilarity = np.reshape(dissimilarity,(np.prod(dissimilarity.shape),))
    
    # return the computed features
    return contrast, energy, homogeneity, correlation, dissimilarity
    
def entropy(labels, base=None):
    # Input arguments
    # labels: array on which compute the entropy
    # base: the base of the logarithm (optional)
    
    # compute the values present in the image along with their occurrance
    value,counts = np.unique(labels, return_counts=True)
    # compute the frequency of each value normalizing by the number of events
    norm_counts = counts / counts.sum()
    # choose the base for the exponential
    base = e if base is None else base
    # return the shannon entropy that is the sum of -p log p
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()
