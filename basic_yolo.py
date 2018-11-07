#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:06:32 2018

@author: dobrowol
"""
import sys
from pathlib import Path
from prepare_trained_model import PrepareTrainedModel, loadFromJson, trainOnData, loadEntireModel
from matplotlib import pyplot as plt
from readrecords import readAnnotations, constructYvector

import context
import numpy as np

preparingModelStrategy = None

context = context.Context()

my_file = Path('./data/model.json')
if my_file.exists():
    text = ""
    while (text != "yes" and text != "no"):
        text = input("previous model exists. load it? (yes|no)")
    if text == "yes" :
        preparingModelStrategy = PrepareTrainedModel(loadFromJson)
    else:
        preparingModelStrategy = PrepareTrainedModel(trainOnData)
else:
    preparingModelStrategy = PrepareTrainedModel(trainOnData)

model = preparingModelStrategy.load(context)

#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)

#(X_train, y_train), (X_test, y_test) = mnist.load_data()
PASopts = {}
PASopts['imgdir']='VOCdata/';

PASopts['resultsdir']='';


PASlabels=[['PASmotorbike','PASmotorbikeSide'],['PASbicycle','PASbicycleSide'],['PASperson','PASpersonSitting','PASpersonStanding','PASpersonWalking'],['PAScar','PAScarFrontal','PAScarRear','PAScarSide']]



model_json = model.to_json()
with open("./data/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("./data/model.h5")
model.save("./data/entire_model.h5")
print("Saved model to disk")


print ("shape of X_test ", len(context.X), " y_test ", len(context.y))
score = model.evaluate(context.X, context.y, verbose=0) 

#print('Accuracy:', score[1])

predictions = model.predict(context.X)

plt.figure(figsize=(15, 15)) 
for i in range(10):    
    ax = plt.subplot(2, 10, i + 1)    
    plt.imshow(context.X[i, :, :, 0], cmap='gray')    
    plt.title("Digit: {}\nPredicted:    {}".format(np.argmax(context.y[i]), np.argmax(predictions[i])))    
    plt.axis('off') 
plt.show()
