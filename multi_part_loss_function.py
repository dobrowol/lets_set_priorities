import keras.backend as K
from math import sqrt
import numpy as np
import tensorflow as tf

def multi_part_loss_function(y_true,y_pred):
    with tf.Session() as sess:   
        tf.initialize_all_variables().run() # need to initialize all variables

        is_cell_responsible=y_true[:,:,0]
        is_box1_responsible=y_true[:,:,1]
        is_box2_responsible=y_true[:,:,6]
    return 5*tf.einsum('ai,aij->aij', is_cell_responsible, K.square(y_pred[:,:,2:3]-y_true[:,:,2:3]))+5*tf.einsum('ai,aij->aij', is_cell_responsible, K.square(sqrt(y_pred[:,:,4:5])-sqrt(y_true[:,:,4:5])))+\
           K.square(y_pred[:,:,11:14]-y_true[:,:,11:14])+0.5*K.square(is_cell_responsible*(y_pred[:,:,0]-y_true[:,:,0]))
