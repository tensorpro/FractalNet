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

    
def fractal_block(incoming, filters, ncols=4, fsize=[3,3],
                  joined=False, reuse=False, scope=None, name="FractalBlock"):
    col = "Col{}".format(ncols)
    with tf.variable_op_scope([incoming], scope, name, reuse=reuse) as scope:
        net = fractal_expand(incoming, filters, ncols, fsize, joined)
    return net
        
def fractal_expand(incoming, filters, ncols, fsize=[3,3], joined=False):
    col = "Col{}".format(ncols)
    left = [tflearn.conv_2d(incoming, filters, fsize, name=col, activation='relu')]
    if(ncols==1):
        return left
    right = join(fractal_expand(incoming, filters, ncols-1, fsize))
    right = fractal_expand(right, filters, ncols-1, fsize)
    out =  left + right
    return join(out) if joined else out

