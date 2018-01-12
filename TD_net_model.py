import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf


def pre_process(img):
    pass

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def next_batch(no_data_points, batch_size):
    pass

class TDNet:

    def __init__(self):
        pass

    def build_model(self):
        pass

class ModelTraining:

    def __init__(self, model, data):
        self.model = model
        self.train_data = data

    def train_model(self):
        pass

class ModelTesting:

    def __init__(self, model, data):
        self.model = model
        self.test_data = data

    def test_model(self):
        pass

if __name__ == '__main__':
    net = TDNet()
