import keras.backend as K
from math import sqrt
import numpy as np
import tensorflow as tf

def multi_part_loss_function(y_true,y_pred):
    is_cell_responsible=y_true[:,:,:,0]
    is_box1_responsible=y_true[:,:,:,1]
    is_box2_responsible=y_true[:,:,:,10]
    print("square 15,18 ", y_true)
    print("y pred 15 18 ", y_pred)
    is_box1_in_cell_responsible = tf.einsum('aij,aij->aij',
        is_cell_responsible,is_box1_responsible)
    is_box2_in_cell_responsible = tf.einsum('aij,aij->aij',
        is_cell_responsible,is_box2_responsible)
    print ("is box cell responsible ", is_box2_in_cell_responsible)
    return 5*K.square(tf.einsum('aij,aijk->aijk', is_box2_in_cell_responsible, 
            y_pred[:,:,:,15:18]-y_true[:,:,:,15:18]))
    #5*K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
    #            y_pred[:,:,2:3,:]-y_true[:,:,2:3,:]))+\
    #        5*K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible, 
    #            sqrt(y_pred[:,:,4:5,:])-sqrt(y_true[:,:,4:5,:])))+\
    #        5*K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
    #            y_pred[:,:,6:9,:]-y_true[:,:,6:9,:]))+\
    #        5*K.square(tf.einsum('aij,aijk->aijk', is_box2_in_cell_responsible,
    #            y_pred[:,:,11:12,:]-y_true[:,:,11:12,:]))+\
    #        5*K.square(tf.einsum('aij,aijk->aijk', is_box2_in_cell_responsible,
    #            sqrt(y_pred[:,:,13:14,:])-sqrt(y_true[:,:,13:14,:])))+\
    #        5*K.square(tf.einsum('aij,aijk->aijk', is_box2_in_cell_responsible, 
    #        y_pred[:,:,15:18,:]-y_true[:,:,15:18,:])) +\
    #        0.5*K.square(is_cell_responsible*(y_pred[:,:,0,:]-y_true[:,:,0,:]))
