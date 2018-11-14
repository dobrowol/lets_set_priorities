#!/usr/bin/env python3

import multi_part_loss_function
import numpy as np
import tensorflow as tf

X = np.zeros((10,10,10,10))

Y = np.ones((10,10,10,10))

res = multi_part_loss_function.multi_part_loss_function(X,Y)

with tf.Session() as sess:
    print (sess.run(res))
