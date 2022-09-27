# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:31:02 2019

@ author: donggue
@ modification : gunwoo lee

"""
import gc
import cv2
import random
import numpy as np
from keras.utils.data_utils import Sequence
#from keras.utils import to_categorical
import FileHandler as FH


class DataGenerator_MultInput3Img_4ItgSteatosisSol_Regression(Sequence):
    def __init__(self, InDataDIR_Bmode, InDataDIR_TAI, InDataDIR_TSI, mode='training', class_size=1, batch_size=1, 
                 ListTxtName=[], resize_shape_Bmode=(640, 480), resize_shape_TAI=(256, 90), resize_shape_TSI=(256, 90), 
                 #horizontal_flip=True, vertical_flip=False, brightness=0.0, rotation=0.0, zoom=0.0, MaskOpt=False):
                 horizontal_flip=True, vertical_flip=False, brightness=0.0, rotation=10.0, zoom=0.05, MaskOpt=False):
       
        DataList, ClassList = FH.readListOfFilePathnClassForTrainNVal_woInPath(InDataDIR_Bmode, ListTxtName)   
        self.mode = mode
        self.MaskOpt = MaskOpt
        self.DataList = DataList
        self.ClassList = ClassList
        self.class_size = class_size
        self.batch_size = batch_size
        
        self.InDataDIR_Bmode = InDataDIR_Bmode
        self.InDataDIR_TAI = InDataDIR_TAI
        self.InDataDIR_TSI = InDataDIR_TSI
        
        self.resize_shape_Bmode = resize_shape_Bmode
        self.resize_shape_TAI = resize_shape_TAI
        self.resize_shape_TSI = resize_shape_TSI
        
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness = brightness
        self.rotation = rotation
        self.zoom = zoom
        
        
        # Preallocate memory
        if self.resize_shape_Bmode:
            self.X_Bmode = np.zeros((batch_size, resize_shape_Bmode[1], resize_shape_Bmode[0], 3), dtype='float32')
        if self.resize_shape_TAI:
            self.X_TAI = np.zeros((batch_size, resize_shape_TAI[1], resize_shape_TAI[0], 3), dtype='float32')
        if self.resize_shape_TSI:
            self.X_TSI = np.zeros((batch_size, resize_shape_TSI[1], resize_shape_TSI[0], 3), dtype='float32')
            
        else:
            raise Exception('No image dimensions specified!')
            
        self.Y = np.zeros((batch_size, class_size), dtype='float32')
        
    def __len__(self):
        return len(self.DataList) // self.batch_size
        
    def __getitem__(self, i):  
        for n, (image_path, GT) in enumerate(zip(self.DataList[i*self.batch_size:(i+1)*self.batch_size], 
                                                    self.ClassList[i*self.batch_size:(i+1)*self.batch_size])):
            
            image_Bmode = cv2.imread(self.InDataDIR_Bmode + "/" + image_path, 1)
            image_TAI = cv2.imread(self.InDataDIR_TAI + "/" + image_path, 1)
            image_TSI = cv2.imread(self.InDataDIR_TSI + "/" + image_path, 1)
            
            if self.resize_shape_Bmode:
                image_Bmode = cv2.resize(image_Bmode, self.resize_shape_Bmode)
                
            if self.resize_shape_TAI:
                image_TAI = cv2.resize(image_TAI, self.resize_shape_TAI)
                
            if self.resize_shape_TAI:
                image_TSI = cv2.resize(image_TSI, self.resize_shape_TAI)
                
            # Do augmentation (only if training)
            if self.mode == 'training':
                if self.horizontal_flip and random.randint(0,1):
                    image_Bmode = cv2.flip(image_Bmode, 1)
                    image_TAI = cv2.flip(image_TAI, 1)
                    image_TSI = cv2.flip(image_TSI, 1)
                if self.vertical_flip and random.randint(0,1):
                    image_Bmode = cv2.flip(image_Bmode, 0)
                    image_TAI = cv2.flip(image_TAI, 0)
                    image_TSI = cv2.flip(image_TSI, 0)
                if self.rotation:
                    angle = random.gauss(mu=0.0, sigma=self.rotation)
                else:
                    angle = 0.0
                if self.zoom:
                    scale = random.gauss(mu=1.0, sigma=self.zoom)
                else:
                    scale = 1.0
                if self.rotation or self.zoom:
                    M = cv2.getRotationMatrix2D((image_Bmode.shape[1]//2, image_Bmode.shape[0]//2), angle, scale)
                    image_Bmode = cv2.warpAffine(image_Bmode, M, (image_Bmode.shape[1], image_Bmode.shape[0]))
                    
                    M = cv2.getRotationMatrix2D((image_TAI.shape[1]//2, image_TAI.shape[0]//2), angle, scale)
                    image_TAI = cv2.warpAffine(image_TAI, M, (image_TAI.shape[1], image_TAI.shape[0]))
                    
                    M = cv2.getRotationMatrix2D((image_TSI.shape[1]//2, image_TSI.shape[0]//2), angle, scale)
                    image_TSI = cv2.warpAffine(image_TSI, M, (image_TSI.shape[1], image_TSI.shape[0]))
                    
                    
            self.X_Bmode[n] = image_Bmode.astype('float32') / 255.
            self.X_TAI[n] = image_TAI.astype('float32') / 255.
            self.X_TSI[n] = image_TSI.astype('float32') / 255.
            self.Y[n, :] = float(GT)
            
            
        return [self.X_Bmode, self.X_TAI, self.X_TSI], self.Y
        
    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(zip(self.DataList, self.ClassList))
        random.shuffle(c)
        self.DataList, self.ClassList = zip(*c)

        # Fix memory leak (Keras bug)
        gc.collect()
    