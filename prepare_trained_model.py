from keras.models import model_from_json, load_model
import keras.backend as K
import numpy as np 
import re

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from readrecords import readAnnotations, constructYvector
from multi_part_loss_function import multi_part_loss_function


from os import path
#from tensorflow.python import debug as tf_debug
#from keras.datasets import mnist
#from oct2py import octave
from collections import defaultdict
from os import path
import re
from skimage import transform 
from scipy import misc

class PrepareTrainedModel(object):
    def __init__(self, load_strategy):
        self._load_strategy = load_strategy
    def load(self, context):
        return self._load_strategy(context)


def prepareData(X, y, records, inputSize):
    print("Loading images...")
    test_images_count = 0
    train_images_count = 0
    print ("record loaded ", len(records))
    for rec in records:
        filename=path.join('VOCdata',rec.imagename)
        train_images_count += 1
        X.append(transform.resize(misc.imread(filename), (inputSize,inputSize)))	
        y.append(constructYvector(rec, inputSize))



    print ("Images loaded ", train_images_count, " X len ", len(X))
    X = np.array(X)
    rows, cols = X[0].shape[0], X[0].shape[1] 
    X = X.reshape(X.shape[0], rows, cols, 3) 
    X = np.vstack(X)
    #X_train = X_train.astype('float32')/255
    #X_test = X_test.astype('float32')/255

    y = np.array(y)

def loadFromJson(context):
    context.records = readAnnotations('./VOCdata/VOC2005_2/Annotations/')
    prepareData(context.X_test, context.y_test, context.records, context.INPUT_SIZE)
    json_file = open('./data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./data/model.h5")
    loaded_model.compile(loss= multi_part_loss_function, optimizer='adam', metrics=['accuracy'])
    print("Loaded model from disk")
    return loaded_model

def loadEntireModel(context):
    context.records = readAnnotations('./VOCdata/VOC2005_2/Annotations/')
    prepareData(context.X_test, context.y_test, context.records, context.INPUT_SIZE)
    loaded_model = load_model("./data/entire_model.h5")
    loaded_model.compile(loss= multi_part_loss_function, optimizer='adam', metrics=['accuracy'])
    print("Loaded model from disk")
    return loaded_model

def trainOnData(context):
    
#FD=octave.extractfeatures(PASopts,imgset)
    context.records = readAnnotations('./VOCdata/VOC2005_1/Annotations/')
    prepareData(context.X_train, context.y_train, context.records, context.INPUT_SIZE)
    context.records = readAnnotations('./VOCdata/VOC2005_2/Annotations/')
    prepareData(context.X_test, context.y_test, context.records, context.INPUT_SIZE)
    print("Start training...")
    print("Y train shape is ", context.y_train.shape)
    print("Y test shape is ", context.y_test.shape)
    #y_train = to_categorical(y_train, num_of_classes) 
    #y_test = to_categorical(y_test, num_of_classes) 

    print("Y train shape is ", context.y_train.shape)
    print("Y test shape is ", context.y_test.shape)

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

    
    model.compile(loss= multi_part_loss_function, optimizer='adam', metrics=['accuracy'])


    model.fit(context.X_train, context.y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.2) 
    return model