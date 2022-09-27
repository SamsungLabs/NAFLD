# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:44:54 2019

@ author: donggue
@ modification : gunwoo lee

"""


from keras import layers
#from keras import Input
from keras.models import Model
#from keras.layers import Lambda
#from keras import backend as K

from keras import applications
#from keras.layers.normalization import BatchNormalization
#from keras import regularizers


 

def build_VGGBased_MultInput3Img_4ItgSteatosisSol_regression(Size_BMode, Size_TAI, Size_TSI, n_classes, nBatch, weights_path=None):
    
    
    VGGNet_Bmode = applications.VGG16(weights='imagenet', include_top=False, input_shape=(Size_BMode[1], Size_BMode[0], 3))
    for i, layer in enumerate(VGGNet_Bmode.layers):
        layer._name = layer._name + "_Bmode"
    x_Bmode = VGGNet_Bmode.output
    x_Bmode = layers.Flatten(name='flatten_Bmode')(x_Bmode)
    x_Bmode = layers.Dense(2048, activation='relu', name='fc1_Bmode')(x_Bmode)
    # x_Bmode = layers.Dense(4096, activation='relu', name='fc1_Bmode')(x_Bmode)
    x_Bmode = layers.Dropout(0.5)(x_Bmode)
    x_Bmode = layers.Dense(1024, activation='relu', name='fc2_Bmode')(x_Bmode)
    x_Bmode = layers.Dropout(0.5)(x_Bmode)
    
    
    VGGNet_TAI = applications.VGG16(weights='imagenet', include_top=False, input_shape=(Size_TAI[1], Size_TAI[0], 3))
    for i, layer in enumerate(VGGNet_TAI.layers):
        layer._name = layer._name + "_TAI"
    x_TAI = VGGNet_TAI.output
    x_TAI = layers.Flatten(name='flatten_TAI')(x_TAI)
    x_TAI = layers.Dense(1024, activation='relu', name='fc1_TAI')(x_TAI)
    x_TAI = layers.Dropout(0.5)(x_TAI)
    x_TAI = layers.Dense(512, activation='relu', name='fc2_TAI')(x_TAI)
    x_TAI = layers.Dropout(0.5)(x_TAI)
    
    VGGNet_TSI = applications.VGG16(weights='imagenet', include_top=False, input_shape=(Size_TSI[1], Size_TSI[0], 3))
    for i, layer in enumerate(VGGNet_TSI.layers):
        layer._name = layer._name + "_TSI"
    x_TSI = VGGNet_TSI.output
    x_TSI = layers.Flatten(name='flatten_TSI')(x_TSI)
    x_TSI = layers.Dense(1024, activation='relu', name='fc1_TSI')(x_TSI)
    x_TSI = layers.Dropout(0.5)(x_TSI)
    x_TSI = layers.Dense(512, activation='relu', name='fc2_TSI')(x_TSI)
    x_TSI = layers.Dropout(0.5)(x_TSI)
    
    x_Concatenate = layers.concatenate([x_Bmode, x_TAI, x_TSI])
    
    x_Concatenate = layers.Dense(512, activation='relu', name='fc3_Concatenate')(x_Concatenate)
    x_Concatenate = layers.Dropout(0.5)(x_Concatenate)
    
    predictions = layers.Dense(n_classes, activation='linear', name='predictions')(x_Concatenate)


    SteatosisModel = Model(inputs=[VGGNet_Bmode.input, VGGNet_TAI.input, VGGNet_TSI.input], outputs=predictions)
    
    SteatosisModel.summary()
        
    return SteatosisModel
