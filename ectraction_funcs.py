# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 17:12:08 2021

@author: ahmed
"""

import numpy as np
import float_to_fb32 as fb
import keras
import cv2
from keras import backend as K
import os
import math

def extract_inputs(conversion_type, image_path, image_name, referance_path, output_path, width, precision):

    test=cv2.imread(f'{image_path}{image_name}.jpg',0)/255.
    test_ref=cv2.imread(f'{referance_path}ref.png',0)/255.0
    if(conversion_type == 0):
        file_name = output_path + "image_" + image_name + "_dec.txt"
    if(conversion_type == 1):
        file_name = output_path + "image_" + image_name + "_fixed.txt"
    if(conversion_type == 2):
        file_name = output_path + "image_" + image_name + "_float.txt"

    test = fb.padding(2, 2, test_ref, test)
    with open(f"{file_name}", "w") as txt_file:
        for filter_row in range(len(test)):
                    for filter_col in range(len(test[filter_row])):
                        if(conversion_type == 0):
                            txt_file.write(f'{test[filter_row][filter_col]}\n')
                        if(conversion_type == 1):
                            txt_file.write(f'{fb.float2fix_complement(test[filter_row][filter_col], width, precision)}\n')
                        if(conversion_type == 2):
                            txt_file.write(f'{fb.binary(test[filter_row][filter_col])}\n')
    
                    
    
    
def ectract_outputs(conversion_type, image_path, image_name, output_path, width, precision, model):
    
    test=cv2.imread(f'{image_path}{image_name}.jpg',0)/255.
    test = test.reshape(1,28,28,1)
    for i in range(len(model.layers)):
        
        file_name = output_path + "output_"+str(i)+"_layer_"+image_name+".txt"
        inp = model.input
        outputs = [layer.output for layer in model.layers]          
        functors = [K.function([inp], [outputs[i]]) ][0](test)  
        the_wanted_functors = np.array(functors)
        if i > 8:
            the_wanted_functors = the_wanted_functors.reshape(1, 1, the_wanted_functors.shape[0], the_wanted_functors.shape[1], the_wanted_functors.shape[2])
        #print(the_wanted_functors.shape)
        with open(f"{file_name}", "w") as txt_file:
            #txt_file.write(f'{w}\n')
            
            
            for filter_row in range(len(the_wanted_functors)):
                for filter_col in range(len(the_wanted_functors[filter_row])):
                    for a_count in range((the_wanted_functors.shape[4])):
                        channel_arr = the_wanted_functors[filter_row][filter_col]
                        for channel in range(len(channel_arr)):
                            for weight in range(len(channel_arr[channel])):
                                #print("\t\t\tweight ",weight)
                                w_arr = channel_arr[channel][weight]
                                w = w_arr[a_count]
                                if(conversion_type == 0):
                                    txt_file.write(f'{w}\n')
                                if(conversion_type == 1):
                                    txt_file.write(f'{fb.float2fix_complement(w, width, precision)}\n')
                                if(conversion_type == 2):
                                    txt_file.write(f'{fb.binary(w)}\n')
            
                    
def ectract_weights(conversion_type, units, output_path, width, precision, model):
    fc_count = 0
    ab_flag = 0 #0 ia a, 1 is b
    layer_counter = 1
    for i in range(len(model.layers)):
        weights=np.array(model.layers[i].get_weights(),dtype="object")
        if ((model.layers[i].name.find("conv") != -1) | (model.layers[i].name.find("dense") != -1)):
            if model.layers[i].name.find("conv") != -1:
                if ab_flag % 2 == 0: # conva
                    extend_zeros = math.ceil(weights[0].shape[2] / units[layer_counter-1])
                    extend_zeros = extend_zeros  * units[layer_counter-1]
                    weight_arr = weights[0]
                    bias_arr = weights[1]
                    for unit in range(units[layer_counter-1]):
                        file_name = f'{output_path}layer_{layer_counter}_mem_{unit}.txt'
                        if os.path.exists(f"{file_name}"):
                            os.remove(f"{file_name}")
                    for weight in range(weights[0].shape[3]):
                        for channel in range(extend_zeros):
                            for unit in range(units[layer_counter-1]):
                                if(channel % units[layer_counter-1] == unit):
                                    file_name = f'{output_path}layer_{layer_counter}_mem_{channel % units[layer_counter-1]}.txt'
                                    with open(f"{file_name}", "a") as txt_file:
                                        for filter_row in range(len(weight_arr)):
                                            for filter_col in range(len(weight_arr[filter_row])):
                                                channel_arr = weight_arr[filter_row][filter_col]
                                                if(channel > weights[0].shape[2]-1 ):
                                                    w = 0
                                                else: 
                                                    w = channel_arr[channel][weight]
                                                if(conversion_type == 0):
                                                    w_fb = w
                                                if(conversion_type == 1):
                                                    w_fb = fb.float2fix_complement(w, width, precision)
                                                if(conversion_type == 2):
                                                    w_fb = fb.binary(w)
                                                txt_file.write(f'{w_fb}\n')
                                        
                    file_name = f'{output_path}layer_{layer_counter}_mem_bias.txt'
                    with open(f"{file_name}", "w") as txt_file:
                        for bias in range(len(bias_arr)):
                            b = bias_arr[bias]
                            if(conversion_type == 0):
                                b_fb = b
                            if(conversion_type == 1):
                                b_fb = fb.float2fix_complement(b, width, precision)
                            if(conversion_type == 2):
                                b_fb = fb.binary(b)
                            txt_file.write(f'{b_fb}\n')                                    
                else:
                    extend_zeros = math.ceil(weights[0].shape[3] / units[layer_counter-1])
                    extend_zeros = extend_zeros  * units[layer_counter-1]
                    weight_arr = weights[0]
                    bias_arr = weights[1]
                    for unit in range(units[layer_counter-1]):
                        file_name = f'{output_path}layer_{layer_counter}_mem_{unit}.txt'
                        if os.path.exists(f"{file_name}"):
                            os.remove(f"{file_name}")
                    for channel in range(weights[0].shape[2]):
                        for weight in range(extend_zeros):
                            for unit in range(units[layer_counter-1]):
                                if(weight % units[layer_counter-1] == unit):
                                    file_name = f'{output_path}layer_{layer_counter}_mem_{weight % units[layer_counter-1]}.txt'
                                    with open(f"{file_name}", "a") as txt_file:
                                        for filter_row in range(len(weight_arr)):
                                            for filter_col in range(len(weight_arr[filter_row])):
                                                channel_arr = weight_arr[filter_row][filter_col]
                                                if(weight > weights[0].shape[3]-1 ):
                                                    w = 0
                                                else: 
                                                    w = channel_arr[channel][weight]
                                                if(conversion_type == 0):
                                                    w_fb = w
                                                if(conversion_type == 1):
                                                    w_fb = fb.float2fix_complement(w, width, precision)
                                                if(conversion_type == 2):
                                                    w_fb = fb.binary(w)
                                                txt_file.write(f'{w_fb}\n')
                        file_name_bias = f'{output_path}layer_{layer_counter}_mem_bias.txt'
                        with open(f"{file_name_bias}", "w") as txt_file_bias:
                            bias_len = math.ceil(len(bias_arr) / units[layer_counter-1])
                            bias_len = bias_len * units[layer_counter-1]  
                            for unit in range(units[layer_counter-1]):
                                file_name = f'{output_path}layer_{layer_counter}_mem_bias_{unit}.txt'
                                if os.path.exists(f"{file_name}"):
                                    os.remove(f"{file_name}")
                            for bias in range(bias_len):
                                for unit in range(units[layer_counter-1]):
                                    if (bias % units[layer_counter-1] == unit):
                                        file_name = f'{output_path}layer_{layer_counter}_mem_bias_{unit}.txt'
                                        with open(f"{file_name}", "a") as txt_file:
                                            if(bias > len(bias_arr)-1 ):
                                                b = 0
                                            else: 
                                                b = bias_arr[bias]
                                            if(conversion_type == 0):
                                                b_fb = b
                                            if(conversion_type == 1):
                                                b_fb = fb.float2fix_complement(b, width, precision)
                                            if(conversion_type == 2):
                                                b_fb = fb.binary(b)
                                            txt_file.write(f'{b_fb}\n')
                        
                            for unit in range(units[layer_counter-1]):
                                file_name = f'{output_path}layer_{layer_counter}_mem_bias_{unit}.txt'
                                with open(f"{file_name}", "r") as txt_file:
                                    for line in txt_file:
                                        txt_file_bias.write(f'{line.rstrip()}\n')
                                        
                ab_flag = ab_flag + 1
            elif  model.layers[i].name.find("dense") != -1:
                fc_count = fc_count + 1
                if(i == 13):
                    weights[1] = np.reshape(weights[1], (1 ,1 , weights[1].shape[0], weights[1].shape[1]))
                    weight_arr = weights[1]
                    bias_arr = weights[0]
                    weight_range = weights[1].shape[3]
                else:
                    weights[0] = np.reshape(weights[0], (1 ,1 , weights[0].shape[0], weights[0].shape[1]))
                    weight_arr = weights[0]
                    bias_arr = weights[1]
                    weight_range = weights[0].shape[3]
                file_name_bias = f'{output_path}layer_{layer_counter}_mem.txt'
                with open(f"{file_name_bias}", "w") as txt_file_bias:
                    for weight in range(weight_range):
                        file_name = f'{output_path}layer_{layer_counter}_mem_{weight}.txt'
                        with open(f"{file_name}", "w") as txt_file:
                            for filter_row in range(len(weight_arr)):
                                for filter_col in range(len(weight_arr[filter_row])):
                                    channel_arr = weight_arr[filter_row][filter_col]
                                    for channel in range(len(channel_arr)):
                                        w = channel_arr[channel][weight]
                                        if(conversion_type == 0):
                                            w_fb = w
                                        if(conversion_type == 1):
                                            w_fb = fb.float2fix_complement(w, width, precision)
                                        if(conversion_type == 2):
                                            w_fb = fb.binary(w)
                                        txt_file.write(f'{w_fb}\n')
                                        txt_file_bias.write(f'{w_fb}\n')
            
                #print("\tbias ")   
                file_name = f'{output_path}layer_{layer_counter}_mem_bias.txt'
                with open(f"{file_name}", "w") as txt_file:
                    for bias in range(len(bias_arr)):
                        b = bias_arr[bias]
                        if(conversion_type == 0):
                            b_fb = b
                        if(conversion_type == 1):
                            b_fb = fb.float2fix_complement(b, width, precision)
                        if(conversion_type == 2):
                            b_fb = fb.binary(b)
                        #reg force {{top.uut.FC{fc_count}.DP.FIFO1.fifo_data_out_{bias+1}}} 32\'b   ; 
                        txt_file.write(f'reg force {{top.uut.FC{fc_count}.DP.FIFO1.fifo_data_out_{bias+1}}} 32\'{b_fb};\n')
                                            
            layer_counter = layer_counter + 1             
                        
        
                             
                            
            
    
    