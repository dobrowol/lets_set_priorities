import keras.backend as K
import tensorflow as tf

def multi_part_loss_function(y_true,y_pred):
    is_cell_responsible=y_true[:,:,:,0]
    is_box1_responsible=y_true[:,:,:,1]
    #is_box2_responsible=y_true[:,:,:,10]
    
    is_box1_in_cell_responsible = tf.einsum('aij,aij->aij',
        is_cell_responsible,is_box1_responsible)
    #is_box2_in_cell_responsible = tf.einsum('aij,aij->aij',
    #    is_cell_responsible,is_box2_responsible)
    
    part1 = 5* tf.einsum("aijk->aij",K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
                                   y_pred[:, :, :, 2:4] - y_true[:, :, :, 2:4])))
    part2 = 5*tf.einsum("aijk->aij",K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
            tf.sqrt(y_pred[:,:,:,4:6])-tf.sqrt(y_true[:,:,:,4:6]))))
    part3 = 5*K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
        y_pred[:,:,:,6:10]-y_true[:,:,:,6:10]))
  #  part4 = 5*K.square(tf.einsum('aij,aijk->aijk', is_box2_in_cell_responsible,
  #      y_pred[:,:,:,11:12]-y_true[:,:,:,11:12]))
  #  part5 = 5*K.square(tf.einsum('aij,aijk->aijk', is_box2_in_cell_responsible,
  #      tf.sqrt(y_pred[:,:,:,13:14])-tf.sqrt(y_true[:,:,:,13:14])))
  #  part6 = 5*K.square(tf.einsum('aij,aijk->aijk', is_box2_in_cell_responsible, 
  #      y_pred[:,:,:,15:18]-y_true[:,:,:,15:18]))

    #part7 = 0.5*K.square(tf.einsum('aij,aij->aij',is_cell_responsible,(y_pred[:,:,:,0]-y_true[:,:,:,0])))
    return K.square(y_true - y_pred)#part1+part2#+part3+part4+part5+part6#+part7