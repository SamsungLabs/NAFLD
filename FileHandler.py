# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:51:10 2019

@ author: donggue
@ modification : gunwoo lee

"""
#import os
#import random
#import cv2

def readListOfFilePathnClassForTrainNVal_woInPath(TxtPath, ListTxtName):
    
    FileName = []
    Class = []
    for i in range(0, len(ListTxtName)):
        Path = TxtPath + "/" + ListTxtName[i] + ".txt"
        File = open(Path, 'r')
        
        while(1):
            line = File.readline()
            
            try: escape = line.index('\n')
            except: escape = len(line)
            
            if line:
                Splt = line[0:escape].split(" ")
                FileName.append(Splt[0])
                Class.append(Splt[1])
            else:
                break
            
        File.close()
        
    
    return FileName, Class    

