#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:06:32 2018

@author: dobrowol
"""

import numpy as np 
import re
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from readrecords import readAnnotations, constructYvector
from scipy import misc
from skimage import transform 
from os import path

from keras.datasets import mnist
#from oct2py import octave
from collections import defaultdict

#(X_train, y_train), (X_test, y_test) = mnist.load_data()
PASopts = {}
PASopts['imgdir']='VOCdata/';

PASopts['resultsdir']='';

Labels = ['VOCmotorbikes', 'VOCbicycles', 'VOCpeople', 'VOCcars']
PASlabels=[['PASmotorbike','PASmotorbikeSide'],['PASbicycle','PASbicycleSide'],['PASperson','PASpersonSitting','PASpersonStanding','PASpersonWalking'],['PAScar','PAScarFrontal','PAScarRear','PAScarSide']]

records = readAnnotations()
#FD=octave.extractfeatures(PASopts,imgset)

INPUT_SIZE=448
X_train=[]
X_test=[]
y_train=[]
y_test=[]
print("Loading images...")
for rec in records:
    filename=path.join('VOCdata',rec.imagename)
    if re.match("test", rec.imagename, re.IGNORECASE):
        X_test.append(transform.resize(misc.imread(filename), (INPUT_SIZE,INPUT_SIZE)))	
        y_test.append(constructYvector(rec, INPUT_SIZE))
    else:
        X_train.append(transform.resize(misc.imread(filename), (INPUT_SIZE,INPUT_SIZE)))	
        y_train.append(constructYvector(rec, INPUT_SIZE))
print("Start training...")
X_train = np.array(X_train)
X_test = np.array(X_test)
rows, cols = X_train[0].shape[0], X_train[0].shape[1] 
X_train = X_train.reshape(X_train.shape[0], rows, cols, 3) 
X_test = X_test.reshape(X_test.shape[0], rows, cols, 3) 

#X_train = X_train.astype('float32')/255
#X_test = X_test.astype('float32')/255

num_of_classes = len(Labels) 
y_train = np.array(y_train)
y_test = np.array(y_test)
print("Y train shape is ", y_train.shape)
print("Y test shape is ", y_test.shape)
y_train = to_categorical(y_train, num_of_classes) 
y_test = to_categorical(y_test, num_of_classes) 

print("Y train shape is ", y_train.shape)
print("Y test shape is ", y_test.shape)

model = Sequential() 
#==========model 1.0==================
#model.add(Conv2D(64, kernel_size=(7, 7), strides=2, input_shape=(rows, cols, 3)))
#model.add(LeakyReLU())
#model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
#model.add(Conv2D(192, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
#model.add(Conv2D(128, kernel_size=(1, 1))) 
#model.add(LeakyReLU())
#model.add(Conv2D(256, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.add(Conv2D(256, kernel_size=(1, 1))) 
#model.add(LeakyReLU())
#model.add(Conv2D(512, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
#model.add(Conv2D(256, kernel_size=(1, 1))) 
#model.add(LeakyReLU())
#model.add(Conv2D(512, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.add(Conv2D(256, kernel_size=(1, 1))) 
#model.add(LeakyReLU())
#model.add(Conv2D(512, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.add(Conv2D(256, kernel_size=(1, 1))) 
#model.add(LeakyReLU())
#model.add(Conv2D(512, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.add(Conv2D(256, kernel_size=(1, 1))) 
#model.add(LeakyReLU())
#model.add(Conv2D(512, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.add(Conv2D(512, kernel_size=(1, 1))) 
#model.add(LeakyReLU())
#model.add(Conv2D(1024, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
#model.add(Conv2D(512, kernel_size=(1, 1))) 
#model.add(LeakyReLU())
#model.summary()
#print("----------------------")
#model.add(Conv2D(1024, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.summary()
#print("----------------------")
#model.add(Conv2D(1024, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.summary()
#print("----------------------")
#model.add(Conv2D(1024, kernel_size=(3, 3), strides=2))#20 layer 
#model.add(LeakyReLU())
#model.summary()
#print("----------------------")
#model.add(Conv2D(1024, kernel_size=(3, 3))) #21 layer
#model.add(LeakyReLU())
#model.summary()
#print("----------------------")
#model.add(Conv2D(1024, kernel_size=(3, 3))) 
#model.add(LeakyReLU())
#model.summary()
#print("----------------------")
#model.add(Conv2D(9, kernel_size=(1, 1))) 
#model.summary()
print("----------------------")
#============end of model 1.0=================

#model.add(Conv2D(52, kernel_size=(3, 3), activation='relu')) 
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Conv2D(26, kernel_size=(3, 3), activation='relu')) 
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Conv2D(13, kernel_size=(3, 3), activation='relu')) 
#model.add(MaxPooling2D(pool_size = (2, 2)))
#model.add(Conv2D(13, kernel_size=(3, 3), activation='relu')) 
#model.add(Conv2D(13, kernel_size=(3, 3), activation='relu')) 
#model.add(Conv2D(13, kernel_size=(1, 1), activation='relu')) 


#model.add(Dropout(0.5)) 



#model.add(Flatten())

#model.add(Dense(128, activation='relu')) 
#model.add(Dropout(0.5)) 
#model.add(Dense(num_of_classes, activation='softmax'))


#============model yolo 2.0===============

print ("imput size ", rows, " ", cols)
model.add(Conv2D(32, kernel_size=(3, 3), padding="same", strides=1, input_shape=(rows, cols, 3)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
model.add(Conv2D(64, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(64, kernel_size=(1, 1), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(128, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(128, kernel_size=(1, 1), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(256, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(256, kernel_size=(1, 1), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(256, kernel_size=(1, 1), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(512, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size = (2, 2), strides=2))

model.add(Conv2D(1024, kernel_size=(3, 3), padding="same", strides=1))#layer 15
model.add(LeakyReLU())
model.add(Conv2D(512, kernel_size=(1, 1), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(1024, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(512, kernel_size=(1, 1), padding="same", strides=1))
model.add(LeakyReLU())
model.add(Conv2D(1024, kernel_size=(3, 3), padding="same", strides=1))
model.add(LeakyReLU())
#model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
#
#
#model.add(Conv2D(1024, kernel_size=(3, 3), padding="same", strides=1))
#model.add(LeakyReLU())
#model.add(Conv2D(1024, kernel_size=(3, 3), padding="same", strides=1))
#model.add(LeakyReLU())
#
#model.add(Conv2D(1024, kernel_size=(3, 3), padding="same", strides=1))
#model.add(LeakyReLU())
model.add(Conv2D(19, kernel_size=(1, 1), strides=1))
model.add(LeakyReLU())
model.summary()

from multi_part_loss_function import multi_part_loss_function
model.compile(loss= multi_part_loss_function, optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2) 

score = model.evaluate(X_test, y_test, verbose=0) 

print('Accuracy:', score[1])

predictions = model.predict(X_test)

plt.figure(figsize=(15, 15)) 
for i in range(10):    
    ax = plt.subplot(2, 10, i + 1)    
    plt.imshow(X_test[i, :, :, 0], cmap='gray')    
    plt.title("Digit: {}\nPredicted:    {}".format(np.argmax(y_test[i]), np.argmax(predictions[i])))    
    plt.axis('off') 
plt.show()
