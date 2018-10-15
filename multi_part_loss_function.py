import keras.backend as K
from math import sqrt
import numpy as np
import tensorflow as tf

def multi_part_loss_function(y_true,y_pred):
    is_cell_responsible=np.multiply(y_pred[:,:,0],y_true[:,:,0])
    
    return 5*K.square(y_pred[:,:,1:2]-y_true[:,:,1:2])+5*K.square(sqrt(y_pred[:,:,3:4])-sqrt(y_true[:,:,3:4]))+\
           K.square(y_pred[:,:,5:8]-y_true[:,:,5:8])+0.5*K.square(is_cell_responsible*(y_pred[:,:,0]-y_true[:,:,0]))
