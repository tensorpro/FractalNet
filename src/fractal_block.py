from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import tflearn
from tensorflow import transpose as T
from tensorflow import mul

def tensor_shape(t):
    return t.get_shape().as_list()

def join(t, keep_prob = .85):
    if len(t)==1:
        return t[0]
    with tf.name_scope("Join"):
        size = len(t)-1
        joined=tf.reduce_mean(t,0)
        out = tf.convert_to_tensor(t)
        drop_mask = tf.to_float(tf.concat(0,[[1],tf.random_uniform([size])])>keep_prob)
        masked = T(mul(T(out),drop_mask))
        dropped = tf.reduce_sum(masked)/tf.reduce_sum(drop_mask)
        tf.cond(tflearn.get_training_mode(), lambda: dropped, lambda: joined)
        return joined

    
def fractal_block(incoming, filters, ncols=3, fsize=[3,3],
                  joined=True, reuse=False, scope=None, name="FractalBlock"):

    Ws = [[] for _ in range(ncols)]
    bs = [[] for _ in range(ncols)]

    def conv_block(incoming, col):
        conv = tflearn.conv_2d(incoming, filters, fsize, weights_init ='xavier',
                               activation='relu',
                               name='Col{}'.format(col))
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


    def random_col(cols):
        with tf.variable_op_scope(cols, "RandomColumn"):
            col_idx = tf.random_uniform([],0,len(cols),'int32')
            return tf.gather(cols, col_idx)
    
    def seperated(name="Seperated"):
        with tf.name_scope(name) as scope:
            sep = [incoming] * ncols
            for col in range(ncols):
                with tf.name_scope("Column_{}".format(col)):
                    for W, b in zip(Ws[col], bs[col]):
                        with tf.variable_op_scope([incoming], None, "ConvBlock"):
                            sep[col] = tf.nn.relu(tf.nn.conv2d(sep[col], W, [1,1,1,1], 'SAME') + b)
            return random_col(sep)

    is_training = tflearn.get_training_mode()
    
    with tf.variable_op_scope([incoming], scope, name, reuse=reuse) as scope:
        fractal = together()
        columns = seperated()
        with tf.variable_op_scope([incoming],"DropPath"):
            global_drop = tf.logical_and(is_training, tf.random_uniform([])>.5)
            net = tf.cond(global_drop, lambda: columns, lambda: fractal, name="DropPath")
    return net

# remember tf.mul
