# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:50:35 2019
@ author: donggue
@ modification : gunwoo lee

"""
from keras import backend as K
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from keras import metrics

import Model
import LearningRateFunction as LR
import LoadImage4Train as LIMG

import os
from contextlib import redirect_stdout



class Train_3InImg_SteatosisSol_reg:
    def __init__(self, iGPU, DataDIR_Bmode, DataDIR_TAI, DataDIR_TSI, WeightSaveDIR, nEpoch, ImgShape_Bmode, ImgShape_TAI, ImgShape_TSI, nClass, nBatchSize, 
                 Opt_lrDecay, LearningRate, gamma, decay, LR_StepSize, optim, Momentum, loss, TrainList, ValidList):
        
        self.iGPU = iGPU
        self.DataDIR_Bmode = DataDIR_Bmode
        self.DataDIR_TAI = DataDIR_TAI
        self.DataDIR_TSI = DataDIR_TSI
        self.WeightSaveDIR = WeightSaveDIR
        self.nEpoch = nEpoch
        self.ImgShape_Bmode = ImgShape_Bmode
        self.ImgShape_TAI = ImgShape_TAI
        self.ImgShape_TSI = ImgShape_TSI
        self.nCls = nClass
        self.nBatch = nBatchSize
        
        self.optim = optim
        self.Momentum = Momentum
        self.loss = loss
        
        self.TrainList = TrainList
        self.ValidList = ValidList
        
        self.LearningRate = LearningRate
        self.decay = decay
        self.Opt_lrDecay = Opt_lrDecay
        
        if self.Opt_lrDecay == True:
            self.gamma = gamma
            self.LR_StepSize = LR_StepSize


        if not os.path.exists(self.WeightSaveDIR):
            os.makedirs(self.WeightSaveDIR)

            
    def PrintCondition(self):
        
        lCmt = []
        
        f = open(self.WeightSaveDIR + "/TrainConditionInfo.txt", mode='at', encoding='utf-8')
        
        lCmt.append("Image size(Bmode) = " + str(self.ImgShape_Bmode[0]) + ", " + str(self.ImgShape_Bmode[1]) + "\n")
        lCmt.append("Image size(TAI) = " + str(self.ImgShape_TAI[0]) + ", " + str(self.ImgShape_TAI[1]) + "\n")
        lCmt.append("Image size(TSI) = " + str(self.ImgShape_TSI[0]) + ", " + str(self.ImgShape_TSI[1]) + "\n")
        lCmt.append("nClass = " + str(self.nCls) + "\n")
        lCmt.append("nBatchSize = " + str(self.nBatch) + "\n")
        lCmt.append("nEpoch = " + str(self.nEpoch) + "\n")
        
        lCmt.append("optim = " + str(self.optim) + "\n")
        lCmt.append("Momentum = " + str(self.Momentum) + "\n")
        lCmt.append("loss = " + str(self.loss) + "\n")
        
        lCmt.append("LearningRate = " + str(self.LearningRate) + "\n")
        lCmt.append("Opt_lrDecay = " + str(self.Opt_lrDecay) + "\n")
        lCmt.append("decay = " + str(self.decay) + "\n")
        if self.Opt_lrDecay == True:
            lCmt.append("gamma = " + str(self.gamma) + "\n")
            lCmt.append("LR_StepSize = " + str(self.LR_StepSize) + "\n")
        
        lCmt.append("TrainDataset = " + str(self.TrainList) + "\n")
        lCmt.append("ValidDataset = " + str(self.ValidList) + "\n")
        
        for i in range(0, len(lCmt)):
            f.write(lCmt[i])
        
        f.close

    def DoTraining_MultInput3Img_4ItgSteatosisSol_regression(self):         
        setGPU = '/gpu:' + str(self.iGPU)
                
        with K.tf.device(setGPU):
            config = tf.ConfigProto() #config = tf.compat.v1.ConfigProto #
            #config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)
            
            # Callbacks
            checkpoint = ModelCheckpoint(self.WeightSaveDIR + "/" + "weights.{epoch:03d}-{val_loss:.3f}.h5", monitor='val_loss', mode='min', save_best_only=True)
            tensorboard = TensorBoard(batch_size = self.nBatch)
            
        
            train_generator = LIMG.DataGenerator_MultInput3Img_4ItgSteatosisSol_Regression(
                InDataDIR_Bmode=self.DataDIR_Bmode, InDataDIR_TAI=self.DataDIR_TAI, InDataDIR_TSI=self.DataDIR_TSI, class_size=self.nCls, 
                batch_size=self.nBatch, ListTxtName = self.TrainList, 
                resize_shape_Bmode = self.ImgShape_Bmode, resize_shape_TAI = self.ImgShape_TAI, resize_shape_TSI = self.ImgShape_TSI)
            
            val_generator = LIMG.DataGenerator_MultInput3Img_4ItgSteatosisSol_Regression(
                InDataDIR_Bmode=self.DataDIR_Bmode, InDataDIR_TAI=self.DataDIR_TAI, InDataDIR_TSI=self.DataDIR_TSI, class_size=self.nCls, 
                mode='validation', batch_size=1, ListTxtName = self.ValidList,
                resize_shape_Bmode = self.ImgShape_Bmode, resize_shape_TAI = self.ImgShape_TAI, resize_shape_TSI = self.ImgShape_TSI)
            
            if self.Opt_lrDecay:
                nEpochs_dropStep = self.LR_StepSize / (len(train_generator.X) * len(train_generator))
                lr_decay = LearningRateScheduler(LR.StepDecay(self.LearningRate, self.gamma, nEpochs_dropStep).scheduler)
            else:
                lr_decay = 0
                
                
            # model build
            DLNet = Model.build_VGGBased_MultInput3Img_4ItgSteatosisSol_regression(self.ImgShape_Bmode, self.ImgShape_TAI, self.ImgShape_TSI, self.nCls, self.nBatch, weights_path=None)
            
            with open(self.WeightSaveDIR + '/ModelInfo.txt', 'w') as f:
                with redirect_stdout(f):
                    DLNet.summary()
            
            # Optimizer
            if self.optim == 'SGD':
                optim = optimizers.SGD(lr = self.LearningRate, momentum = self.Momentum, decay = self.decay, nesterov=True)
            else:
                optim = self.optim
            
            if self.loss == 'mse':
                mtrs = metrics.mse
            elif self.loss == 'categorical_crossentropy':
                mtrs = 'categorical_accuracy'
                
            
            # Training
            DLNet.compile(optimizer = optim, loss = self.loss, metrics=[mtrs])
            
            if self.Opt_lrDecay:
                history = DLNet.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=self.nEpoch, 
                                               validation_data=val_generator, validation_steps=len(val_generator), shuffle=True, callbacks=[checkpoint, tensorboard, lr_decay])
            else:
                history = DLNet.fit_generator(train_generator, steps_per_epoch=len(train_generator), epochs=self.nEpoch, 
                                               validation_data=val_generator, validation_steps=len(val_generator), shuffle=True, callbacks=[checkpoint, tensorboard])
           
            return history
