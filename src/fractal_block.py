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
        with tf.variable_op_scope([incoming], None, name, reuse=reuse) as scope:
        # with tf.name_scope(name) as scope:
            out = fractal_expand(incoming)
            net=join(out) if joined else out
        return net

    
    def seperated(name="Seperated"):
        # with tf.variable_op_scope([incoming], scope, name, reuse=reuse) as scope:
        with tf.name_scope(name) as scope:
            sep = [incoming] * ncols
            for col in range(ncols):
                with tf.name_scope("Column_{}".format(col)):
                    for W, b in zip(Ws[col], bs[col]):
                        sep[col] = tf.nn.relu(tf.nn.conv2d(sep[col], W, [1,1,1,1], 'SAME') + b)
            sep = join(sep)
        return sep

    is_training = tflearn.get_training_mode()
    
    with tf.variable_op_scope([incoming], scope, name, reuse=reuse) as scope:
        fractal = together()
        columns = seperated()
        with tf.variable_op_scope([incoming],"DropPath"):
            global_drop = tf.logical_and(is_training, tf.random_uniform([1])[0]>.5)
            net = tf.cond(global_drop, lambda: columns, lambda: fractal, name="DropPath")
    return net

# remember tf.mul
