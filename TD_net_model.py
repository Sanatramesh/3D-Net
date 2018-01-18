import os
# import cv2
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET

DISP_FILE_TYPE = 'xml'

def read_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = root[0][3].text.replace('\n', '')
    data = data.replace('   ', '')
    l = data.strip().split(' ')
    disp_data = np.float32(np.array(l)).reshape(int(root[0][0].text), int(root[0][1].text))
    return disp_data

def read_disparity_map(disp_file):
    global DISP_FILE_TYPE

    if DISP_FILE_TYPE == 'xml':
        disp_map = read_xml(disp_file)
    else:
        disp_map = read_img(disp_file)

    return disp_map

def read_img(img_file):
    img = mpimg.imread(img_file)
    return img

def get_files_in_dir(dir_path):
    if os.path.exists(dir_path):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        files.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
        files = [os.path.join(dir_path, f) for f in files]
        return files

    return None

def get_subdir_in_dir(dir_path):
    if os.path.exists(dir_path):
        dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        return dirs

    return None

def pre_process(img):
    pass

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable( initial )

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable( initial )

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def conv2d_2x2(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 2, 2, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def next_batch(no_data_points, batch_size):
    points = np.random.choice( no_data_points, batch_size )
    return points

def read_data(data_files):
    data = [[],[],[]]

    for left_cam, right_cam, disp_map in data_files:
        data[0].append(read_img(left_cam))
        data[1].append(read_img(right_cam))
        data[2].append(read_disparity_map(disp_map))
        # data.append((read_img(left_cam), read_img(right_cam), read_disparity_map(disp_map)))

    return data



class TDNet:

    def __init__(self, input_dims, output_dims, learning_rate = 1e-4):
        self.y = None                                # Output of the Network
        self.y_ = None                               # Actual Output
        self.x_left, self.x_right = None, None       # Input left and right images
        self.feat_left, self.feat_right = None, None # Extracted features for left and right images
        self.sess = None                             # Tensorflow session
        self.loss, self.train_step = None, None
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate

    def build_model(self):
        pass

    def loss_func(self):
        self.loss = tf.reduce_sum( tf.squared_difference( self.y, self.y_ ) )
        self.train_step = tf.train.AdamOptimizer( self.learning_rate ).minimize( self.loss )

    def sess_init(self):
        self.sess.run( tf.global_variables_initializer() )

    def train_epoch(self, data):
        left_cam_data, right_cam_data, disp_data = data
        _, epoch_loss = self.sess.run( [self.train_step, self.loss],
                                    feed_dict = {self.x_left: left_cam_data,
                                                self.x_right: right_cam_data,
                                                self.y_: disp_data} )
        return epoch_loss

    def forward_pass(self, data):
        disparity_map = self.sess.run( self.y,
                                    feed_dict = {self.x_left: left_cam_data,
                                                self.x_right: right_cam_data} )

        return disparity_map

    def save_model(self, ckpt_file):
        tf.train.Saver( tf.global_variables() ).save( self.sess, ckpt_file )

    def load_model(self, ckpt_file):
        if ckpt_file != None:
            tf.train.Saver().restore( self.sess, ckpt_file )
        else:
            self.sess.run( tf.global_variables_initializer() )

class TDNet_VGG11(TDNet):

    def __init__(self, input_dims, output_dims, learning_rate = 1e-4, vgg_file):
        super().__init__( input_dims, output_dims, learning_rate )
        self.vgg_file = vgg_file

    def build_model(self):
        pass



class ModelTraining:

    def __init__(self, model, data_files, batch_size = 10, epochs = 20):
        self.model = model
        self.train_data_files = data_files # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.batch_size = batch_size
        self.no_epochs = epochs
        self.no_data = len( self.train_data_files )

    def train_model(self):
        # shuffle and read a batch from the train dataset
        # train model for one epoch - call fn model.train_epoch(data)
        print ( 'Training Model ... ' )
        for i in range( self.no_epochs ):
            batch_idx = next_batch( self.no_data, self.batch_size )
            data = read_data( self.train_data_files[ batch_idx ] )
            self.model.train_epoch( data )
            # validate the model and print test, validation accuracy

        print ( 'Training Model ... Complete' )

    def get_model(self):
        return self.model



class ModelTesting:

    def __init__(self, model, data_files):
        self.model = model
        self.test_data_files = data_files # List of tuple: (left_cam, right_cam, disp_map) filenames

    def test_model(self):
        # shuffle and read a batch from the test dataset
        # test model for one epoch - call fn model.forward_pass(data)
        pass

if __name__ == '__main__':
    net = TDNet()
    read_xml('data/NTSD-200/groundtruth/depth_maps/left/frame_1.xml')
    read_img('data/NTSD-200/groundtruth/disparity_maps/left/frame_1.png')
    # print (get_files_in_dir('data/NTSD-200/groundtruth/depth_maps/left/'))
    # print (get_subdir_in_dir('data/NTSD-200/groundtruth/depth_maps/'))
