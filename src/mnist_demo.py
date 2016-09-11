from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from fractal_block import fractal_conv2d


# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
from tensorflow.contrib import slim
# Building convolutional network
net = input_data(shape=[None, 28, 28, 1], name='input')

# filters = [32,64,128,256]
filters = [4,8]
for f in filters:
  net = fractal_conv2d(net, 4, f,3)
  net = slim.max_pool2d(net,2)


net = fully_connected(net, 10, activation='softmax')
net = regression(net, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')


# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

