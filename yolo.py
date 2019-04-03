import tensorflow as tf
import context
from readrecords import readAnnotations
from prepare_trained_model import prepareData
import keras.backend as K
# declare two symbolic floating-point scalars
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
context = context.Context()
context.records = readAnnotations('./VOCdata/VOC2005_1/Annotations/')
prepareData(context)
y_true = tf.convert_to_tensor(context.y)
y_pred = tf.zeros([1, 7, 7, 10], tf.float64)
is_cell_responsible = tf.convert_to_tensor(context.y[:, :, :, 0])
is_box1_responsible = tf.convert_to_tensor(context.y[:, :, :, 1])
print(y_true.get_shape())
# is_box2_responsible=y_true[:,:,:,10]

is_box1_in_cell_responsible = tf.einsum('aij,aij->aij',
                                        is_cell_responsible, is_box1_responsible)
#is_box2_in_cell_responsible = tf.einsum('aij,aij->aij',
#                                        is_cell_responsible, is_box2_responsible)

part1 = 5 * K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
                               y_pred[:, :, :, 2:3] - y_true[:, :, :, 2:3]))
#part2 = 5 * K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
#                               tf.sqrt(y_pred[:, :, :, 4:5]) - tf.sqrt(y_true[:, :, :, 4:5])))
#part3 = 5 * K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
#                               y_pred[:, :, :, 6:9] - y_true[:, :, :, 6:9]))
# create a simple symbolic expression using the add function
add = tf.add(a, b)
# bind 1.5 to ' a ' , 2.5 to ' b ' , and evaluate ' c '
sess = tf.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(is_box1_in_cell_responsible, feed_dict=binding)
print(c)