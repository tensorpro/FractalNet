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
                  joined=True, reuse=False, scope=None, name="FractalBlock"):
    
    def conv_block(incoming, col):
        net = tflearn.conv_2d(incoming, filters, fsize,
                              name='Col{}'.format(col), activation='relu')
        return net
    def fractal_expand(incoming, col=0):
        left = [conv_block(incoming, col)]
        if(col==ncols-1):
            return left
        right = join(fractal_expand(incoming, col+1))
        right = fractal_expand(right, col+1)
        out =  left + right
        return out

    with tf.variable_op_scope([incoming], scope, name, reuse=reuse) as scope:
        out = fractal_expand(incoming)
        return join(out) if joined else out
