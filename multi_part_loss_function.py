import keras.backend as K
import tensorflow as tf

def multi_part_loss_function(y_true,y_pred):
    is_cell_responsible=y_true[:, :, :, 0]
    is_box1_responsible=y_true[:, :, :, 1]
    
    #is_box1_in_cell_responsible = tf.einsum('aij,aij->aij',
    #    is_cell_responsible,is_box1_responsible)
    #is_box2_in_cell_responsible = tf.einsum('aij,aij->aij',
    #    is_cell_responsible,is_box2_responsible)

    y_pred_part1 = y_pred[:, :, :, 0:4]
    y_pred_part2 = tf.sqrt(y_pred[:, :, :, 4:6])
    y_pred_part3 = y_pred[:, :, :, 6:10]
    y_pred_prepared = tf.concat([y_pred_part1, y_pred_part2, y_pred_part3], -1)

    y_true_part1 = y_true[:, :, :, 0:4]
    y_true_part2 = tf.sqrt(y_true[:, :, :, 4:6])
    y_true_part3 = y_true[:, :, :, 6:10]
    y_true_prepared = tf.concat([y_true_part1, y_true_part2, y_true_part3], -1)

    return K.square(tf.einsum('aij,aijk->aijk', is_cell_responsible,
                    y_true_prepared - y_pred_prepared))