from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import tflearn

def tensor_shape(t):
    return t.get_shape().as_list()

def join(t):
    with tf.name_scope("Join"):
        joined=tflearn.dropout(tf.reduce_mean(t,0), .5, name="Join")
        return joined

def fractal_block(incoming, filters, ncols=4, fsize=[3,3]):
    scope = "Conv{}".format(ncols)
    left = tflearn.conv_2d(incoming, filters, fsize, name=scope)
    if(ncols==1):
        return left
    right_1 = fractal_block(incoming, filters, ncols-1, fsize)
    right_2 = fractal_block(right_1, filters, ncols-1, fsize)
    return join([left, right_2])
