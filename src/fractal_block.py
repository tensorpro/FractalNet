from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
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

def fractal_block(inputs,
                  block,
                  num_cols,
                  joined=True,
                  reuse=False,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  scope=None):
  
  def fractal_expand(inputs, block, num_cols, joined):
    '''Recursive Helper Function for making fractal'''
    print(num_cols)
    
    if num_cols == 1:
      out = block(inputs)
      return out if joined else [out]
    
    left = block(inputs)
    right = fractal_expand(inputs, block, num_cols-1, joined=True)
    right = fractal_expand(right, block, num_cols-1, joined=False)
    cols=[left]
    cols.extend(right)
    out = join(cols) if joined else cols
    print(out)
    return out

  with tf.variable_op_scope([inputs], scope, 'Fractal',
                            reuse=reuse) as scope:
    net=fractal_expand(inputs, block, num_cols, joined)
  return net


def wrap_fn(fn,*args,**kwargs):
  return lambda inputs: fn(inputs,*args,**kwargs)
