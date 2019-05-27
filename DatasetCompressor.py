# -*- coding: utf-8 -*-
"""
Created on Sun May 26 00:24:25 2019

@author: anant singh

Dataset Taken from Kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/downloads/chest-xray-pneumonia.zip/2#chest_xray.zip
"""

import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)
imageSize = 60

imageDir = glob.glob("C:/Users/anant singh/Desktop/Xray_classification/ChestXray_Dataset/train/NORMAL/*.jpeg")
normalTrainArray = [cv2.imread(img,0) for img in imageDir]
normalTrainArray = np.array(normalTrainArray)
print(normalTrainArray.shape[0])
for i in range(0,normalTrainArray.shape[0]):
    normalTrainArray[i] = cv2.resize(normalTrainArray[i],(imageSize,imageSize))
    cv2.imwrite("C:/Users/anant singh/Desktop/Xray_classification/Dataset_compressed/train/NORMAL/"+str(i)+".jpg",normalTrainArray[i])

imageDir = glob.glob("C:/Users/anant singh/Desktop/Xray_classification/ChestXray_Dataset/test/NORMAL/*.jpeg")
normalTestArray = [cv2.imread(img,0) for img in imageDir]
normalTestArray = np.array(normalTestArray)
for i in range(0,normalTestArray.shape[0]):
    normalTestArray[i] = cv2.resize(normalTestArray[i],(imageSize,imageSize))
    cv2.imwrite("C:/Users/anant singh/Desktop/Xray_classification/Dataset_compressed/test/NORMAL/"+str(i)+".jpg",normalTestArray[i])

imageDir = glob.glob("C:/Users/anant singh/Desktop/Xray_classification/ChestXray_Dataset/train/PNEUMONIA/*.jpeg")
pneumoniaTrainArray = [cv2.imread(img,0) for img in imageDir]
pneumoniaTrainArray = np.array(pneumoniaTrainArray)
print(pneumoniaTrainArray.shape[0])
for i in range(0,pneumoniaTrainArray.shape[0]):
    pneumoniaTrainArray[i] = cv2.resize(pneumoniaTrainArray[i],(imageSize,imageSize))
    cv2.imwrite("C:/Users/anant singh/Desktop/Xray_classification/Dataset_compressed/train/PNEUMONIA/"+str(i)+".jpg",pneumoniaTrainArray[i])
    
imageDir = glob.glob("C:/Users/anant singh/Desktop/Xray_classification/ChestXray_Dataset/test/PNEUMONIA/*.jpeg")
pneumoniaTestArray = [cv2.imread(img,0) for img in imageDir]
pneumoniaTestArray = np.array(pneumoniaTestArray)
for i in range(0,pneumoniaTestArray.shape[0]):
    pneumoniaTestArray[i] = cv2.resize(pneumoniaTestArray[i],(imageSize,imageSize))
    cv2.imwrite("C:/Users/anant singh/Desktop/Xray_classification/Dataset_compressed/test/PNEUMONIA/"+str(i)+".jpg",pneumoniaTestArray[i])

#finalTrainImages = np.concatenate((normalTrainArray,pneumoniaTrainArray))
#print(finalTrainImages.shape[0])
