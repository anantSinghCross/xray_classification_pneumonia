# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:28:02 2019

@author: anant singh
"""
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)
imageSize = 60

print("initiating input compilation...\n")
print("starting 'normalTrainArray'...\n")
imageDir = glob.glob("C:/Users/anant singh/Desktop/Xray_classification/Dataset_compressed/train/NORMAL/*.jpg")
normalTrainArray = [cv2.imread(img,0) for img in imageDir]
normalTrainArray = np.array(normalTrainArray)
print(normalTrainArray.shape[0])

print("starting 'normalTestArray'...\n")
imageDir = glob.glob("C:/Users/anant singh/Desktop/Xray_classification/Dataset_compressed/test/NORMAL/*.jpg")
normalTestArray = [cv2.imread(img,0) for img in imageDir]
normalTestArray = np.array(normalTestArray)

print("starting 'pneumoniaTrainArray'...\n")
imageDir = glob.glob("C:/Users/anant singh/Desktop/Xray_classification/Dataset_compressed/train/PNEUMONIA/*.jpg")
pneumoniaTrainArray = [cv2.imread(img,0) for img in imageDir]
pneumoniaTrainArray = np.array(pneumoniaTrainArray)
print(pneumoniaTrainArray.shape[0])
    
print("starting 'pneumoniaTestArray'...\n")
imageDir = glob.glob("C:/Users/anant singh/Desktop/Xray_classification/Dataset_compressed/test/PNEUMONIA/*.jpg")
pneumoniaTestArray = [cv2.imread(img,0) for img in imageDir]
pneumoniaTestArray = np.array(pneumoniaTestArray)

x_train = np.concatenate((normalTrainArray,pneumoniaTrainArray))
print(x_train.shape)

x_test = np.concatenate((normalTestArray,pneumoniaTestArray))

# normalizing the inputs
x_train = x_train/255
x_test = x_test/255 

print(x_test.shape)
print()
print("compiled all the input data\n now creating outputs...\n")


print("creating normal outputs...")
normalTrainOutput = np.full(normalTrainArray.shape[0],0)
normalTestOutput = np.full(normalTestArray.shape[0],0)
print(normalTrainOutput)
print(normalTestOutput)

print("creating pneumonia outputs...")
pneumoniaTrainOutput = np.full(pneumoniaTrainArray.shape[0],1)
pneumoniaTestOutput = np.full(pneumoniaTestArray.shape[0],1)
print(pneumoniaTrainOutput)
print(pneumoniaTestOutput)

# combining the training outputs and testing outputs separately

y_train = np.concatenate((normalTrainOutput,pneumoniaTrainOutput))
y_test = np.concatenate((normalTestOutput,pneumoniaTestOutput))

x_train, y_train = shuffle(x_train,y_train)
x_test, y_test = shuffle(x_test,y_test)

print("outputs created successfully\n")
print("======================================================\n")
print("initiating training...")

# creating the a CNN model
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(1,imageSize,imageSize),data_format='channels_first',activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.2))
model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dense(2,activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train = x_train.reshape(x_train.shape[0], 1,imageSize,imageSize)
x_test = x_test.reshape(x_test.shape[0], 1,imageSize,imageSize)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=20,verbose=1)

scores = model.evaluate(x_test,y_test,verbose =1)
print('Error: %.2f%%'% (100 - 100*scores[1]))

model_json = model.to_json()
with open("trainedModel.json","w") as jsonFile:
    jsonFile.write(model_json)
model.save_weights("modelWeights.h5")
print("Saved model to disk along with the weights")

print("\n  * * *\n   * *\n")








