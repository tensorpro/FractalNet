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

    Ws = [[] for _ in range(ncols)]
    bs = [[] for _ in range(ncols)]

    def conv_block(incoming, col):
        conv = tflearn.conv_2d(incoming, filters, fsize,
                              name='Col{}'.format(col), activation='relu')
        Ws[col].append(conv.W)
        bs[col].append(conv.b)
        return conv

    def fractal_expand(incoming, col=0):
        left = [conv_block(incoming, col)]
        if(col==ncols-1):
            return left
        right = join(fractal_expand(incoming, col+1))
        right = fractal_expand(right, col+1)
        out =  left + right
        return out

    
    def together(name="Fractal"):
        # with tf.variable_op_scope([incoming], scope, name, reuse=reuse) as scope:
        with tf.name_scope(name) as scope:
            out = fractal_expand(incoming)
            net=join(out) if joined else out
        return net

    
    def seperated(name="Columns"):
        # with tf.variable_op_scope([incoming], scope, name, reuse=reuse) as scope:
        with tf.name_scope(name) as scope:
            sep = [incoming] * ncols
            for col in range(ncols):
                for W, b in zip(Ws[col], bs[col]):
                    sep[col] = tf.nn.relu(tf.nn.conv2d(sep[col], W, [1,1,1,1], 'SAME') + b)
            sep = join(sep)
        return sep

    with tf.variable_op_scope([incoming], scope, name, reuse=reuse) as scope:
        fractal = join(fractal_expand(incoming))
        columns = seperated()
        is_training = tflearn.get_training_mode()
        net = tf.cond(is_training, lambda: columns, lambda: fractal)
    return net

# remember tf.mul
