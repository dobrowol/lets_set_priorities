from keras.models import model_from_json, load_model
import keras.backend as K
import numpy as np 
import re
import os

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from readrecords import readAnnotations, constructYvector
from multi_part_loss_function import multi_part_loss_function

from os import path

from skimage import transform 
from scipy import misc


class PrepareTrainedModel(object):
    def __init__(self, load_strategy):
        self._load_strategy = load_strategy

    def load(self, context):
        return self._load_strategy(context)


def prepareData(context):
    print("Loading images...")
    train_images_count = 0
    context.X = []
    context.y = []
    print ("record loaded ", len(context.records))
    for rec in context.records:
        filename=path.join('VOCdata',rec.imagename)
        yolo_filename=path.join('YOLOdata',rec.imagename)
        train_images_count += 1
        x = transform.resize(misc.imread(filename), (context.INPUT_SIZE,context.INPUT_SIZE))
        y = constructYvector(rec, context.INPUT_SIZE, context.OUTPUT_SIZE)
        if not os.path.exists(os.path.dirname(yolo_filename)):
            os.makedirs(os.path.dirname(yolo_filename))
        with open(yolo_filename, 'a') as the_file:
            the_file.write('X='+np.array2string(x)+'\n')
            the_file.write("y="+np.array2string(y)+'\n')

        context.X.append(x)	
        context.y.append(y)



    print ("Images loaded ", train_images_count, " X len ", len(context.X))
    context.X = np.array(context.X)
    context.y = np.array(context.y)
    rows, cols = context.X[0].shape[0], context.X[0].shape[1] 
    context.X = context.X.reshape(context.X.shape[0], rows, cols, 3) 
    #context.X = np.vstack(context.X)
    print ("X shape ", context.X.shape)
    print ("y shape ", context.y.shape)
    #X_train = X_train.astype('float32')/255
    #X_test = X_test.astype('float32')/255

    context.y = np.array(context.y)

def loadFromJson(context_with_test_data):
    context_with_test_data.records = readAnnotations('./VOCdata/VOC2005_2/Annotations/')
    prepareData(context_with_test_data)
    json_file = open('./data/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./data/model.h5")
    loaded_model.compile(loss= multi_part_loss_function, optimizer='adam', metrics=['accuracy'])
    print("Loaded model from disk")
    return loaded_model

def loadEntireModel(context_with_test_data):
    context_with_test_data.records = readAnnotations('./VOCdata/VOC2005_2/Annotations/')
    prepareData(context_with_test_data)
    loaded_model = load_model("./data/entire_model.h5")
    loaded_model.compile(loss= multi_part_loss_function, optimizer='adam', metrics=['accuracy'])
    print("Loaded model from disk")
    return loaded_model

def constructCustomYolo( rows, cols, output_size):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", strides=1, input_shape=(rows, cols, 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", strides=1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", strides=1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", strides=1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", strides=1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", strides=1))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), padding="same", strides=1))
    model.add(LeakyReLU())
    model.add(Conv2D(output_size, kernel_size=(1, 1), strides=1))
    model.add(LeakyReLU())
    return model

def constructYolov2(rows, cols, output_size):
    model = Sequential()
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
    model.add(Conv2D(output_size, kernel_size=(1, 1), strides=1))
    model.add(LeakyReLU())
    return model

def trainOnData(context):
    
#FD=octave.extractfeatures(PASopts,imgset)
    context.records = readAnnotations('./VOCdata/VOC2005_1/Annotations/')
    prepareData(context)
    
    print("Start training...")
    print("Y train shape is ", context.y.shape)
    #y_train = to_categorical(y_train, num_of_classes) 
    #y_test = to_categorical(y_test, num_of_classes) 
    rows, cols = context.X[0].shape[0], context.X[0].shape[1]

    
     
    model = constructCustomYolo(rows, cols, context.OUTPUT_SIZE)

    
    model.summary()
    
    model.compile(loss= multi_part_loss_function, optimizer='adam', metrics=['accuracy'])


    model.fit(context.X, context.y, batch_size=12, epochs=10, verbose=1, validation_split=0.2) 
    context.records = readAnnotations('./VOCdata/VOC2005_2/Annotations/')
    prepareData(context)
    return model