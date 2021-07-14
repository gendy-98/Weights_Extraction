# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 17:04:19 2021

@author: ahmed
"""
#%% imports

import numpy as np
import float_to_fb32 as fb
import keras
import cv2
from keras import backend as K
import os
import math
import ectraction_funcs as ef 
import tensorflow_model_optimization as tfmot

import tensorflow as tf


#%% load model

np.set_printoptions(threshold=np.inf)

with tfmot.quantization.keras.quantize_scope():
    model = keras.models.load_model('..\extraction\lenet_opt.hdf5')


#%% 


for i in range(len(model.layers)):
    weights=np.array(model.layers[i].get_weights(),dtype="object")
    
    
weights=np.array(model.layers[1].get_weights(),dtype="object")
print(model.layers[1].name)
print(weights[0].shape[3])
print()

    
#0 decimal 
#1 fixed
#2 flaot
types = 3 
Full_path = '../extraction/input/'
width_float = 32
precision_float = 27
width_fixed = 20
precision_fixed = 12
image_name = '5273'
width  = width_fixed
precision = precision_fixed

image_array = ['5273', 'img_151', '4979', '4977']
for image_name in image_array:
    for conversion_type in range(types):
        if conversion_type == 1:
            width  = width_fixed
            precision = precision_fixed
        if conversion_type == 2:
            width  = width_float
            precision = precision_float     
            
        ef.extract_inputs(conversion_type, Full_path, image_name, Full_path, Full_path, width, precision_fixed)
        

image_name = '4977'
for conversion_type in range(types):
    if conversion_type == 0:
        output_path = '../extraction/output/dec/'
    if conversion_type == 1:
        output_path = '../extraction/output/fixed/'
        width  = width_fixed
        precision = precision_fixed
    if conversion_type == 2:
        output_path = '../extraction/output/float/'
        width  = width_float
        precision = precision_float     
            
    ef.ectract_outputs(conversion_type, Full_path, image_name, output_path, width, precision, model)



units = [1,3,3,1,1]
for conversion_type in range(types):
    if conversion_type == 0:
        output_path = '../extraction/memory/memory_decimal/'
    if conversion_type == 1:
        output_path = '../extraction/memory/memory_fixed/'
        width  = width_fixed
        precision = precision_fixed
    if conversion_type == 2:
        output_path = '../extraction/memory/memory_float/'
        width  = width_float
        precision = precision_float     

    ef.ectract_weights(conversion_type, units, output_path, width, precision, model)
