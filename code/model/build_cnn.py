"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Build CNN - Skeleton
Build TensorFlow computation graph for convolutional network
Usage: `from model.build_cnn import cnn`
"""

import tensorflow as tf
import math


# TODO: can define helper functions here to build CNN graph
def conv2d(x, in_length, in_channels, out_channels, filter_len=5, stride=2, activation='relu'):
	b = tf.Variable(tf.constant(0.01, shape=[ out_channels ]))
	W = tf.Variable(tf.truncated_normal([filter_len, filter_len, in_channels, out_channels], stddev = 0.1 / tf.sqrt(tf.cast(filter_len * filter_len * in_channels, tf.float32))))
	im = tf.reshape(x, [-1, in_length, in_length, in_channels])
	c = tf.nn.conv2d(im, W, strides=[1, stride,stride, 1], padding='SAME') + b 
	
	if activation == 'tanh':
		c = tf.nn.tanh(c)  
	elif activation == 'sigmoid':
		c = tf.nn.sigmoid(c)
	elif activation == 'relu':
		c = tf.nn.relu(c)  
	elif activation == 'linear':
		c = c
	result = tf.reshape(c, [tf.shape(c)[0],  -1])  
	return result

def buildNet(x, in_size, out_size, activation='relu'):
	b = tf.Variable(tf.constant(0.1, shape=[out_size]))
	W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=1.0 / tf.sqrt(tf.cast(in_size, tf.float32))))
	c = tf.matmul(x, W) + b
	if activation == 'tanh':
		c = tf.nn.tanh(c)
	elif activation == 'sigmoid':
		c = tf.nn.sigmoid(c)
	elif activation == 'relu':
		c = tf.nn.relu(c)
	elif activation == 'linear':
		c = c
	result = tf.reshape(c, [-1, out_size])  
	return result

# def conv2d(x, W, b, strides=2, activation='relu'):
#     x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
#     x = tf.nn.bias_add(x, b)
#     if activation == 'relu':
# 		conv = tf.nn.relu(x)  
# 	elif activation == 'tanh':
# 		conv = tf.nn.tanh(x)  
# 	elif activation == 'sigmoid':
# 		conv = tf.nn.sigmoid(x)
#     return conv

def normalize(x):
    ''' Set mean to 0.0 and standard deviation to 1.0 via affine transform '''
    shifted = x - tf.reduce_mean(x)
    scaled = shifted / tf.sqrt(tf.reduce_mean(tf.multiply(shifted, shifted)))
    return scaled


def cnn():
	''' Convnet '''
	# TODO: build CNN architecture graph
	inputSize = 1024
	classSize = 7
	input_layer = tf.placeholder(tf.float32, shape=[None, inputSize]) 
	act = 'linear'
    #may change to other activation
	linearAct = 'linear'
	c1 = conv2d(input_layer, 32, 1, 16, activation=act) 
	c2 = conv2d(c1, 16, 16, 32, activation=act) 
	c3 = conv2d(c2, 8, 32, 64, activation=act)
	net = buildNet(c3, inputSize, 100, activation=act)  
	result = buildNet(net, 100, classSize, activation=linearAct)
	pred_layer = normalize(result)
	return input_layer, pred_layer
