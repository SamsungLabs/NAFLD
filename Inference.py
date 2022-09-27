# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:50:49 2019

@ author: donggue
@ modification : gunwoo lee

"""
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import FileHandler as FH



class Inference_3InImg:
    def __init__(self, iGPU, TestImgTxtPath_Bmode, TestImgTxtPath_TAI, TestImgTxtPath_TSI, TestImageList, WeightSaveDIR, WeightFile, 
                 ImgShape_Bmode, ImgShape_TAI, ImgShape_TSI, nClass):
        
        self.iGPU = iGPU
        self.TestImgTxtPath_Bmode = TestImgTxtPath_Bmode
        self.TestImgTxtPath_TAI = TestImgTxtPath_TAI
        self.TestImgTxtPath_TSI = TestImgTxtPath_TSI
        self.TestImageList = TestImageList
        self.WeightSaveDIR = WeightSaveDIR
        self.WeightFile = WeightFile
        self.ImgShape_Bmode = ImgShape_Bmode
        self.ImgShape_TAI = ImgShape_TAI
        self.ImgShape_TSI = ImgShape_TSI
        self.nClass = nClass
        
        
    def DoInference_Regression_IntgSteatosis_3ImgIn(self):
                
        setGPU = '/gpu:' + str(self.iGPU)
        WeightFN = self.WeightSaveDIR + "/" + self.WeightFile
            
        with K.tf.device(setGPU):
            #config = tf.compat.v1.ConfigProto #config = tf.ConfigProto()
            #config.gpu_options.allow_growth = True
            #sess = tf.Session(config=config)
            #K.set_session(sess)
        
            
            if WeightFN:
                net = load_model(WeightFN, custom_objects={'tf' : tf})
            else:
                print('No checkpoint specified! Set it with the --checkpoint argument option')
                exit()
            
            
            x_Bmode = np.zeros((1, self.ImgShape_Bmode[0], self.ImgShape_Bmode[1], 3), dtype='float32')
            x_TAI = np.zeros((1, self.ImgShape_TAI[0], self.ImgShape_TAI[1], 3), dtype='float32')
            x_TSI = np.zeros((1, self.ImgShape_TSI[0], self.ImgShape_TSI[1], 3), dtype='float32')
            
            print(x_Bmode.shape)
            print(x_TAI.shape)
            print(x_TSI.shape)
            
            DataList, ClassList = FH.readListOfFilePathnClassForTrainNVal_woInPath(self.TestImgTxtPath_Bmode, self.TestImageList)  
            
            f = open(self.WeightSaveDIR + "/TestResult" + str(self.WeightFile) + ".txt", mode='at', encoding='utf-8')
            
            for i in range(0, len(DataList)):
            
                image_Bmode = cv2.imread(self.TestImgTxtPath_Bmode + "/" + DataList[i], 1)
                image_Bmode = cv2.resize(image_Bmode, (self.ImgShape_Bmode[1], self.ImgShape_Bmode[0]))
                
                image_TAI = cv2.imread(self.TestImgTxtPath_TAI + "/" + DataList[i], 1)
                image_TAI = cv2.resize(image_TAI, (self.ImgShape_TAI[1], self.ImgShape_TAI[0]))
                
                image_TSI = cv2.imread(self.TestImgTxtPath_TSI + "/" + DataList[i], 1)
                image_TSI = cv2.resize(image_TSI, (self.ImgShape_TSI[1], self.ImgShape_TSI[0]))
                
                
                
                x_Bmode[0] = image_Bmode.astype('float32') / 255.
                x_TAI[0] = image_TAI.astype('float32') / 255.
                x_TSI[0] = image_TSI.astype('float32') / 255.
                
                y = net.predict([x_Bmode, x_TAI, x_TSI], batch_size=1)
    
            
                Comment = DataList[i] + " : GS = " + str(ClassList[i]) + " Predict = " + str(y[0][0]) + "\n"
                print(Comment)
                
                Comment_Save = DataList[i] + " : GS = " + str(ClassList[i]) + " Predict = " + str(y[0][0]) + "\n"
                f.write(Comment_Save)
                
    
        f.close
