from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope


from tflearn.datasets import mnist
xflat, _,_,_ = mnist.load_data()
batch=xflat[:16].reshape(-1,28,28,1)
net  = batch

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import transpose as T
from tensorflow import mul
from copy import deepcopy

import tflearn

def tensor_shape(tensor):
  return tensor.get_shape().as_list()

def local_drop(cols, drop_prob=.85):
  size = len(cols)-1
  with tf.variable_op_scope(cols, None, "LocalDropPath"):
    out = tf.to_float(cols)
    drop_mask = tf.to_float(tf.concat(0,[[1],tf.random_uniform([size])])>drop_prob)
    masked = T(mul(T(out),tf.random_shuffle(drop_mask)))
    dropped = tf.reduce_sum(masked,0)/tf.reduce_sum(drop_mask,0)
  return dropped

def join(cols, drop_prob = .15):
  if len(cols)==1:
    return cols[0]
  with tf.variable_op_scope(cols, None, "Join"):
    joined=tf.reduce_mean(cols,0)
    out = tf.cond(tflearn.get_training_mode(),
                  lambda: local_drop(cols, drop_prob), lambda: joined)
  return joined

def fractal_template(inputs,
                     num_columns,
                     block_fn,
                     block_as,
                     joined=True,
                     reuse=False,
                     scope=None):
  """Template for making fractal blocks.

  Given a function and a corresponding arg_scope `fractal_template`
  will build a truncated fractal with `num_columns` columns.
  
  Args:
    inputs: a 4-D tensor  `[batch_size, height, width, channels]`.
    num_columns: integer, the columns in the fractal. 
    block_fn: function to be called within each fractal.
    block_as: A function that returns argscope for `block_fn`.
    joined: boolean, whether the output columns should be joined.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    scope: Optional scope for `variable_scope`.
  """
  def fractal_expand(inputs, num_columns, joined):
    '''Recursive Helper Function for making fractal'''
    with block_as():
      output = lambda cols: join(cols) if joined else cols
      if num_columns == 1:
        return output([block_fn(inputs)])
      left = block_fn(inputs)
      right = fractal_expand(inputs, num_columns-1, joined=True)
      right = fractal_expand(right, num_columns-1, joined=False)
      cols=[left]+right
    return output(cols)

  with tf.variable_op_scope([inputs], scope, 'Fractal',
                            reuse=reuse) as scope:
    net=fractal_expand(inputs, num_columns, joined)
  return net

def fractal_conv2d(inputs,
                   num_columns,
                   num_outputs,
                   kernel_size,
                   joined=True,
                   stride=1,
                   padding='SAME',
                   rate=1,
                   activation_fn=nn.relu,
                   normalizer_fn=slim.batch_norm,
                   normalizer_params=None,
                   weights_initializer=initializers.xavier_initializer(),
                   weights_regularizer=None,
                   biases_initializer=None,
                   biases_regularizer=None,
                   reuse=None,
                   variables_collections=None,
                   outputs_collections=None,
                   trainable=True,
                   scope=None):
  """Builds a fractal block with slim.conv2d.
  
  The fractal will have `num_columns` columns, and have 
  Args:
    inputs: a 4-D tensor  `[batch_size, height, width, channels]`.
    num_columns: integer, the columns in the fractal. 
      
  """
  locs = locals()
  fractal_args = ['inputs','num_columns','joined']
  asc_fn = lambda : slim.arg_scope([slim.conv2d],
                                   **{arg:val for (arg,val) in locs.items()
                                      if arg not in fractal_args})
  return fractal_template(inputs, num_columns, slim.conv2d, asc_fn,
                          joined, reuse, scope)
