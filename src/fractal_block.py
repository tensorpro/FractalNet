from __future__ import print_function, division
import tensorflow as tf
import numpy as np

# sess = tf.InteractiveSession()    

def tensor_shape(t):
    return t.get_shape().as_list()

def join(t):
    return tf.reduce_mean(t,0)
    

def conv_weights(shape):
    """
    Returns a tuple of form: (random weights, biases)
    with the specified shape for weights
    """
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.constant(.1, shape=[shape[-1]]))
    return (W,b)

def frac_weights(columns, base_shape, in_chan, out_chan):
    top_shape = base_shape + [in_chan, out_chan]
    lower_shape = base_shape + [out_chan, out_chan]
    depths = (2**np.arange(columns))[::-1]
    top_layer = [conv_weights(top_shape) for _ in range(columns)]
    lower_layers = [[conv_weights(lower_shape)
                    for _ in range(d)]
                    for d in (depths-1)]
    columns = [np.array([top]+rest)
               for (top, rest) in zip(top_layer, lower_layers)]
    return np.array(columns)

def conv(incoming, W,b):
    return tf.nn.relu(tf.nn.conv2d(incoming, W, [1,1,1,1], 'SAME') + b)

def get_weights(col_weights, idx, j):
    col_idx = zip(col_weights[j], idx[j])
    return np.array([c[i] for (c,i) in col_idx])

def frac_block(incoming, col_weights):
    ncols = len(col_weights)
    pow_2s = 2**np.arange(ncols)
    joins = [i % pow_2s == 0 for i in range(1, 2**(ncols-1)+1)]
    cols = np.repeat(incoming, ncols)
    idx = np.repeat(0, ncols)
    for j in joins:
        inputs = cols[j]
        conv_weights = get_weights(col_weights, idx, j)
        convs = [conv(i,*w) for (i,w) in zip(inputs, conv_weights)]
        output = join(convs)
        cols[j] = output
        idx+=j
    return cols[0]

def dense_block(incoming, out_units = 1024):
    in_shape = tensor_shape(incoming)
    in_units = np.product(in_shape[1:])
    W_shape = [in_units, out_units]
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1))
    b = tf.Variable(tf.constant(.1,shape=[out_units]))
    return tf.matmul(tf.reshape(incoming, [-1, in_units]), W) + b
