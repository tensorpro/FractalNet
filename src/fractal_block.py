from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import tflearn

def tensor_shape(t):
    return t.get_shape().as_list()

def join(t):
    if len(t)==1:
        return t[0]
    with tf.name_scope("Join"):
        joined=tflearn.dropout(tf.reduce_mean(t,0), .5, name="Join")
        return joined

def show(t):
    print(tensor_shape(t))

def fractal_block(incoming, filters, ncols=4, fsize=[3,3], joined=False):
    name = "Conv{}".format(ncols)
    left = [tflearn.conv_2d(incoming, filters, fsize, name=name, activation='relu')]
    if(ncols==1):
        return left
    right_1 = join(fractal_block(incoming, filters, ncols-1, fsize))
    right_2 = fractal_block(right_1, filters, ncols-1, fsize)
    out =  left + right_2
    if joined:
        return join(out)
    else:
        return out
