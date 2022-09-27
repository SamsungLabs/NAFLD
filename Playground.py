# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:39:40 2019

@ author: donggue
@ modification : gunwoo lee

"""
import Train as Tr
import Inference as Inf


RunCmd = 1 # 1 : Train regression, 10 : Inference regression

iGPU = 0
 
if  RunCmd == 1:
    
    nEpoch = 1500
    iGPU = 0
    nBatchSize = 32
    LearningRate = 1E-06

   
    RegTrain = Tr.Train_3InImg_SteatosisSol_reg(iGPU, 
                                                  r"D:\Liver\3_Data\Train_data\Bmode", 
                                                  r"D:\Liver\3_Data\Train_data\TAI_PNG",
                                                  r"D:\Liver\3_Data\Train_data\TSI_PNG",  
                                                  r"D:\Liver\5_Exp\DLSteatosisTest_Val\IntegSol\DLSteatosis_IntegSol", 
                                                  nEpoch, (320, 240), (256, 90), (256, 90), 1, nBatchSize, False, LearningRate, 0.1, LearningRate/2, 200000, 'SGD', 0.9, 'mse', 
                                                  ["train_GT_Img_QUSExistAll"], ["val_GT_Img_QUSExistAll"])
    Tr.Train_3InImg_SteatosisSol_reg.PrintCondition(RegTrain)   
    Hist = Tr.Train_3InImg_SteatosisSol_reg.DoTraining_MultInput3Img_4ItgSteatosisSol_regression(RegTrain)
    
        
elif RunCmd == 10:
    
      
    RegInf = Inf.Inference_3InImg(iGPU, r"D:\Liver\3_Data\Test_data\Bmode", 
                                        r"D:\Liver\3_Data\Test_data\TAI_PNG",
                                        r"D:\Liver\3_Data\Test_data\TSI_PNG", 
                                       ["test_GT_Img_QUSExistAll"], r"D:\Liver5_Exp\DLSteatosisTest_Val\IntegSol\DLSteatosis_IntegSol", 
                                       "weights.249-9.780.h5", (240, 320), (90, 256), (90, 256), 1)
    Inf.Inference_3InImg.DoInference_Regression_IntgSteatosis_3ImgIn(RegInf)
    

    

        