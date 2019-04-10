import tensorflow as tf
import context
from readrecords import readAnnotations
from prepare_trained_model import prepareData
import keras.backend as K
import numpy as np
import pytest

context = context.Context()
context.records = readAnnotations('./VOCdata/VOC2005_1/Test_Annotations/')
prepareData(context)
y_true = tf.convert_to_tensor(context.y)
y_pred = tf.zeros([1, 7, 7, 10], tf.float64)
is_cell_responsible = tf.convert_to_tensor(context.y[:, :, :, 0])
is_box1_responsible = tf.convert_to_tensor(context.y[:, :, :, 1])

# is_box2_responsible=y_true[:,:,:,10]

is_box1_in_cell_responsible = tf.einsum('aij,aij->aij',
                                        is_cell_responsible, is_box1_responsible)
#is_box2_in_cell_responsible = tf.einsum('aij,aij->aij',
#                                        is_cell_responsible, is_box2_responsible)

square_root_of_predicted_weights_and_heights = tf.sqrt(y_pred[:, :, :, 4:6])
square_root_of_expected_weights_and_heights = tf.sqrt(y_true[:, :, :, 4:6])

#part2 = 5 * K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
#                               tf.sqrt(y_pred[:, :, :, 4:5]) - tf.sqrt(y_true[:, :, :, 4:5])))
#part3 = 5 * K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
#                               y_pred[:, :, :, 6:9] - y_true[:, :, :, 6:9]))
# create a simple symbolic expression using the add function

# bind 1.5 to ' a ' , 2.5 to ' b ' , and evaluate ' c '


def test_selection_of_box_with_center():
    sess = tf.Session()
    c = sess.run(is_box1_in_cell_responsible)

    expected = np.zeros([1, 7, 7])
    expected[0, 1, 1] = expected[0, 1, 3] = expected[0, 3, 3] = 1
    assert((c == expected).all())


def test_difference_of_coordinates():
    sess = tf.Session()
    c = sess.run(y_pred[:, :, :, 2:4] - y_true[:, :, :, 2:4])

    assert(pytest.approx(c[0, 1, 1, 0], 0.001) == -0.948)
    assert (pytest.approx(c[0, 1, 1, 1], 0.001) == -0.996)
    assert (pytest.approx(c[0, 3, 3, 1], 0.01) == -0.593)
    assert (pytest.approx(c[0, 3, 3, 0], 0.01) == -0.301)


def test_select_only_important_coordinates():
    y_pred = tf.ones([1, 7, 7, 10], tf.float64)
    part1 =  tf.einsum("aijk->aij",K.square(tf.einsum('aij,aijk->aijk', is_box1_in_cell_responsible,
                                   y_pred[:, :, :, 2:4] - y_true[:, :, :, 2:4] )))
    sess = tf.Session()
    c = sess.run(part1)
    expected = np.square(1-0.948) + np.square(1-0.996)
    expected_too = np.square(1 - 0.593) + np.square(1 - 0.301)

    assert (pytest.approx(c[0, 1, 1], 0.1) == expected)
    assert (pytest.approx(c[0, 3, 3], 0.01) == expected_too)


def test_multipart_function():
    y_pred = tf.constant(4.0, shape=[1, 7, 7, 10], dtype="float64")
    y_pred_part1 = y_pred[:, :, :, 0:4]
    y_pred_part2 = tf.sqrt(y_pred[:, :, :, 4:6])
    y_pred_part3 = y_pred[:, :, :, 6:10]
    y_pred_prepared = tf.concat([y_pred_part1, y_pred_part2, y_pred_part3], -1)

    y_true_part1 = y_true[:, :, :, 0:4]
    y_true_part2 = tf.sqrt(y_true[:, :, :, 4:6])
    y_true_part3 = y_true[:, :, :, 6:10]
    y_true_prepared = tf.concat([y_true_part1, y_true_part2, y_true_part3], -1)

    sess = tf.Session()
    c = sess.run(y_pred_prepared)
    expected = np.square(1-0.948) + np.square(1-0.996)
    expected_too = np.square(1 - 0.593) + np.square(1 - 0.301)
    print(c)
    print(c.shape)

    sess = tf.Session()
    c = sess.run(y_true_prepared)
    print(c)
    print(c.shape)

    multipart1 = K.square(tf.einsum('aij,aijk->aijk', is_cell_responsible,
                       y_pred - y_true))

