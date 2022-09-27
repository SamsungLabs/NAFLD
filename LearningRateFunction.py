# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:25:06 2019

@author: donggue
"""
import math
import numpy as np
    
class StepDecay:
    def __init__(self, initial_lr, power, n_epochs, decay, nEpochs_dropStep):
        self.initial_lr = initial_lr
        self.power = power
        self.n_epochs = n_epochs
        self.decay = decay
        self.nEpochs_dropStep = nEpochs_dropStep
    
    def scheduler(self, epoch):
        return self.initial_lr * math.pow(self.decay, math.floor((1 + epoch) / self.nEpochs_dropStep))
    
class PolyDecay:
    def __init__(self, initial_lr, power, n_epochs):
        self.initial_lr = initial_lr
        self.power = power
        self.n_epochs = n_epochs
    
    def scheduler(self, epoch):
        return self.initial_lr * np.power(1.0 - 1.0*epoch/self.n_epochs, self.power)