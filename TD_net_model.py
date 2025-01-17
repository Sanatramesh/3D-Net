import os
# import cv2
import sys
import time
import datetime
import argparse
import numpy as np
import tensorflow as tf
from random import shuffle
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
from scipy.misc import imread, imsave

import vgg16_encoder, vgg16_encoder2

# DISP_FILE_TYPE = 'PNG_SINTEL'
# LEFT_CAM_DIR  = 'data/MPI-Sintel-stereo-training-20150305/training/clean_left'
# RIGHT_CAM_DIR = 'data/MPI-Sintel-stereo-training-20150305/training/clean_right'
# DISPARITY_DIR = 'data/MPI-Sintel-stereo-training-20150305/training/disparities'

DISP_FILE_TYPE = 'PNG_NTSD'
CAM_DIR  = 'data/NTSD_complete/illumination'
DISPARITY_DIR = 'data/NTSD_complete/groundtruth/disparity_maps'
IMG_DIMS = (480, 640)

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

    if DISP_FILE_TYPE == 'XML':
        disp_map = read_xml(disp_file)
    elif DISP_FILE_TYPE == 'PNG_SINTEL':
        # Disparity read function inherited from Sintel sdk
        f_in = read_img(disp_file)
        d_r = f_in[:,:,0].astype('float64')
        d_g = f_in[:,:,1].astype('float64')
        d_b = f_in[:,:,2].astype('float64')
        disp_map = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    else:
        f_in = imread(disp_file)
        disp_map = f_in.astype('float64')

    disp_shp = disp_map.shape
    disp_map = disp_map.reshape((disp_shp[0], disp_shp[1], 1))

    return disp_map

def read_img(img_file):
    img = imread(img_file)
    return img[:,:,:3]

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
        dirs.sort()
        # dirs = [os.path.join(dir_path, d) for d in dirs]
        return dirs

    return None

def read_data(data_files):
    data = [[],[],[]]

    for left_cam, right_cam, disp_map in data_files:
        data[0].append(read_img(left_cam))
        data[1].append(read_img(right_cam))
        data[2].append(read_disparity_map(disp_map))
        # data.append((read_img(left_cam), read_img(right_cam), read_disparity_map(disp_map)))

    return data

# Read func for Sintel dataset
def get_data_files_SINTEL(left_dir, right_dir, disp_dir):
    left_imgs, right_imgs, disp_imgs = [], [], []

    for d in get_subdir_in_dir(left_dir):
        left_imgs  += get_files_in_dir(os.path.join(left_dir, d))
        right_imgs += get_files_in_dir(os.path.join(right_dir, d))
        disp_imgs  += get_files_in_dir(os.path.join(disp_dir, d))

    return list(zip(left_imgs, right_imgs, disp_imgs))

# Read func for NTSD-200 dataset
def get_data_files_NTSD(cam_dir, disp_dir):
    left_imgs, right_imgs, disp_imgs = [], [], []

    for d in get_subdir_in_dir(cam_dir):
        left_imgs  += get_files_in_dir(os.path.join(cam_dir, d, 'left'))
        right_imgs += get_files_in_dir(os.path.join(cam_dir, d, 'right'))
        disp_imgs  += get_files_in_dir(os.path.join(disp_dir, 'left'))

    return list(zip(left_imgs, right_imgs, disp_imgs))

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

def relu_activation(input_tensor, name = 'relu'):
    output_tensor = tf.nn.relu(input_tensor, name = name)
    return output_tensor

def convtranspose2d_layer(input_tensor, num_filters, kernel_size, strides, padding, name = 'upsample'):
    output_tensor = tf.layers.conv2d_transpose(inputs = input_tensor, filters = num_filters,
                        kernel_size = kernel_size, strides = strides, padding = padding, name = name)
    return output_tensor

def conv2d_layer(input_tensor, num_filters, kernel_size, strides, padding, name = 'conv'):
    output_tensor = tf.layers.conv2d(inputs = input_tensor, filters = num_filters,
                        kernel_size = kernel_size, strides = strides, padding = padding, name = name)
    return output_tensor

def next_batch(no_data_points, batch_size):
    points = np.random.choice( no_data_points, batch_size )
    return list(points)


class TDNet(object):

    def __init__(self, input_dims, output_dims, learning_rate = 1e-4):
        self.y = None                            # Output of the Network
        self.y_ = None                         # Actual Output
        self.x_left, self.x_right = None, None       # Input left and right images
        self.feat_left, self.feat_right = None, None # Extracted features for left and right images
        self.sess = None                             # Tensorflow session
        self.feat_cmb = None
        self.encoder_pl, self.decoder_pl = None, None
        self.loss, self.train_step = None, None
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate

    def build_model(self):
        dmap_shape = [None] + self.output_dims + [1]
        self.y_ = tf.placeholder(tf.float32, shape = dmap_shape)
        encoder_pl = tf.placeholder(tf.float32, shape = encoder_shape)

        self.build_decoder()

    def build_encoder(self):
        pass

    def feature_combine(self):
        # Multiply features
        self.feat_cmb = tf.multiply(self.feat_left, self.feat_right)
        # Add features
        self.feat_cmb = tf.add(self.feat_left, self.feat_right)

    def build_decoder(self):
        dec1_1 = conv2d_layer( self.decoder_pl, num_filters = 256, kernel_size = [3, 3],
                                    strides = [1, 1], padding = 'same', name = 'conv4' )
        dec1_2 = relu_activation( dec1_1, name = 'relu4' )
        dec1_3 = convtranspose2d_layer( dec1_2, num_filters = 256, kernel_size = [4, 4],
                                    strides = [2, 2], padding = 'same', name = 'upsample1' )


        dec2_1 = conv2d_layer( dec1_3, num_filters = 128, kernel_size = [3, 3],
                                    strides = [1, 1], padding = 'same', name = 'conv5' )
        dec2_2 = relu_activation( dec2_1, name = 'relu5' )
        dec2_3 = convtranspose2d_layer( dec2_2, num_filters = 128, kernel_size = [4, 4],
                                    strides = [2, 2], padding = 'same', name = 'upsample2' )


        dec3_1 = conv2d_layer( dec2_3, num_filters = 64, kernel_size = [3, 3],
                                    strides = [1, 1], padding = 'same', name = 'conv6' )
        dec3_2 = relu_activation( dec3_1, name = 'relu6' )
        dec3_3 = convtranspose2d_layer( dec3_2, num_filters = 64, kernel_size = [4, 4],
                                    strides = [2, 2], padding = 'same', name = 'upsample3' )

        out = conv2d_layer( dec3_3, num_filters = 1, kernel_size = [3, 3],
                                    strides = [1, 1], padding = 'same', name = 'conv7' )
        self.y = relu_activation( out, name = 'relu7' )

    def add_loss_optimizer(self):
        filtx = tf.constant(np.array([[0,0,0],[-0.5,0,0.5], [0,0,0]], dtype = np.float32).reshape((3,3,1,1)), name="filter")
        # filty = tf.constant(np.array([[0,-0.5,0],[0,0,0], [0,0.5,0]], dtype = np.float32).reshape((3,3,1,1)), name="filter")
        filty = tf.transpose(filtx, [1, 0, 2, 3])

        dyx  = tf.nn.conv2d(self.y, filtx, [1, 1, 1, 1], padding='SAME')
        dyy  = tf.nn.conv2d(self.y, filty, [1, 1, 1, 1], padding='SAME')

        dy_x  = tf.nn.conv2d(self.y_, filtx, [1, 1, 1, 1], padding='SAME')
        dy_y  = tf.nn.conv2d(self.y_, filty, [1, 1, 1, 1], padding='SAME')

        self.loss = tf.reduce_mean( tf.squared_difference( self.y, self.y_ ) )
        self.loss += tf.reduce_sum( tf.squared_difference( dyx, dy_x ) )
        self.loss += tf.reduce_sum( tf.squared_difference( dyy, dy_y ) )
        self.train_step = tf.train.AdamOptimizer( self.learning_rate ).minimize( self.loss )

    def sess_init(self):
        self.sess = tf.Session()
        self.sess.run( tf.global_variables_initializer() )

    def train_epoch(self, data):
        left_cam_data, right_cam_data, disp_data = data
        _ = self.sess.run( self.train_step,
                            feed_dict = {self.x_left: left_cam_data,
                                        self.x_right: right_cam_data,
                                        self.y_: disp_data} )

    def compute_loss(self, data):
        left_cam_data, right_cam_data, disp_data = data
        model_loss = self.sess.run( self.loss,
                            feed_dict = {self.x_left: left_cam_data,
                                        self.x_right: right_cam_data,
                                        self.y_: disp_data} )
        return model_loss

    def forward_pass(self, data):
        left_cam_data, right_cam_data, disp_data = data
        disparity_map = self.sess.run( self.y,
                            feed_dict = {self.x_left: left_cam_data,
                                        self.x_right: right_cam_data} )

        return disparity_map

    def save_model(self, ckpt_file):
        tf.train.Saver( tf.global_variables() ).save( self.sess, ckpt_file )

    def load_model(self, ckpt_file = None):
        if ckpt_file != None:
            tf.train.Saver().restore( self.sess, ckpt_file )
        else:
            self.sess.run( tf.global_variables_initializer() )

    def get_name(self):
        return 'TDNet with custom encoder'


class TDNet_VGG11(TDNet):

    def __init__(self, input_dims, output_dims, decoder_shape, vgg_file, learning_rate = 1e-4):
        super(TDNet_VGG11, self).__init__( input_dims, output_dims, learning_rate )
        self.vgg = None
        self.vgg_file = vgg_file
        self.decoder_shape = decoder_shape

    def build_model(self):
        encoder_shape = [None] + self.input_dims
        self.decoder_shape = [None] + self.decoder_shape
        dmap_shape = [None] + self.output_dims

        self.y_ = tf.placeholder(tf.float32, shape = dmap_shape)
        self.encoder_pl = tf.placeholder(tf.float32, shape = encoder_shape)

        self.vgg = vgg16_encoder.Vgg16( self.vgg_file )
        with tf.name_scope( "content_vgg" ):
            self.vgg.build( self.encoder_pl )
        self.decoder_pl =  tf.placeholder(tf.float32, shape = self.decoder_shape)
        self.build_decoder()

    def train_batch(self, data):
        left_cam_data, right_cam_data, disp_data = data
        # get the feature maps for left camera images
        feat_left  = self.sess.run( self.vgg.pool3, feed_dict = {self.encoder_pl : left_cam_data} )
        # get the feature maps for right camera images
        feat_right = self.sess.run( self.vgg.pool3, feed_dict = {self.encoder_pl : right_cam_data} )
        # combine feature maps
        feature_maps = self.feature_combine( feat_left, feat_right )

        _ = self.sess.run( self.train_step,
                            feed_dict = {self.decoder_pl: feature_maps,
                                        self.y_: disp_data} )

    def compute_loss(self, data):
        left_cam_data, right_cam_data, disp_data = data
        # get the feature maps for left camera images
        feat_left  = self.sess.run( self.vgg.pool3, feed_dict = {self.encoder_pl : left_cam_data} )
        # get the feature maps for right camera images
        feat_right = self.sess.run( self.vgg.pool3, feed_dict = {self.encoder_pl : right_cam_data} )
        # combine feature maps
        feature_maps = self.feature_combine( feat_left, feat_right )

        model_loss = self.sess.run( self.loss,
                            feed_dict = { self.decoder_pl: feature_maps,
                                        self.y_: disp_data } )
        return model_loss

    def forward_pass(self, data):
        left_cam_data, right_cam_data, disp_data = data
        # get the feature maps for left camera images
        feat_left  = self.sess.run( self.vgg.pool3, feed_dict = {self.encoder_pl : left_cam_data} )
        # get the feature maps for right camera images
        feat_right = self.sess.run( self.vgg.pool3, feed_dict = {self.encoder_pl : right_cam_data} )
        # combine feature maps
        feature_maps = self.feature_combine( feat_left, feat_right )

        disparity_map = self.sess.run( self.y,
                            feed_dict = { self.decoder_pl: feature_maps } )

        return disparity_map

    def feature_combine(self, left_feat, right_feat):
        global IMG_DIMS

        # Multiply features
        # feat_combined = np.multiply( left_feat, right_feat )

        # Add features
        # feat_combined = np.add( left_feat, right_feat )

        # Subtract features
        feat_combined = np.subtract( left_feat, right_feat )

        # Interweaving features
        # shp = list(left_feat.shape)
        # shp[-1] = shp[-1] * 2
        # shp = tuple(shp)
        #
        # feat_combined = np.empty(shp, dtype = left_feat.dtype)

        # if shp[0] == IMG_DIMS[0]:
        #     feat_combined[:,:,0::2] = left_feat
        #     feat_combined[:,:,1::2] = right_feat
        # else:
        #     feat_combined[:,:,:,0::2] = left_feat
        #     feat_combined[:,:,:,1::2] = right_feat

        # Concat features
        # feat_combined = np.concatenate( ( left_feat, right_feat ), axis = 3 )

        return feat_combined

    def get_name(self):
        return 'TDNet with VGG11 encoder'

class TDNet_VGG11_V2(TDNet):

    def __init__(self, input_dims, output_dims, vgg_file, learning_rate = 1e-4):
        super(TDNet_VGG11_V2, self).__init__( input_dims, output_dims, learning_rate )
        self.vgg = None
        self.vgg_file = vgg_file

    def build_model(self):
        encoder_shape = [None] + self.input_dims
        dmap_shape = [None] + self.output_dims

        self.y_ = tf.placeholder(tf.float32, shape = dmap_shape)
        self.x_left  = tf.placeholder(tf.float32, shape = encoder_shape)
        self.x_right = tf.placeholder(tf.float32, shape = encoder_shape)

        self.vgg = vgg16_encoder2.Vgg16( self.vgg_file )
        with tf.name_scope( "content_vgg" ):
            self.vgg.build( self.x_left, self.x_right )

        self.feature_combine()
        self.decoder_pl = self.feat_cmb
        self.build_decoder()

    def feature_combine(self):
        # Multiply features
        # self.feat_cmb = tf.multiply(self.vgg.pool3_l, self.vgg.pool3_r)

        # Add features
        #self.feat_cmb = tf.add(self.vgg.pool3_l, self.vgg.pool3_r)

        # Subtract features
        self.feat_cmb = tf.subtract(self.vgg.pool3_l, self.vgg.pool3_r)

        # Concat features
        # self.feat_cmb = tf.concat([self.vgg.pool3_l, self.vgg.pool3_r], axis = 3)

    def train_batch(self, data):
        left_cam_data, right_cam_data, disp_data = data

        _ = self.sess.run( self.train_step,
                            feed_dict = {self.x_left: left_cam_data,
                                        self.x_right: right_cam_data,
                                        self.y_: disp_data} )

    def compute_loss(self, data):
        left_cam_data, right_cam_data, disp_data = data

        model_loss = self.sess.run( self.loss,
                            feed_dict = {self.x_left: left_cam_data,
                                        self.x_right: right_cam_data,
                                        self.y_: disp_data} )
        return model_loss

    def forward_pass(self, data):
        left_cam_data, right_cam_data, disp_data = data

        disparity_map = self.sess.run( self.y,
                            feed_dict = {self.x_left: left_cam_data,
                                        self.x_right: right_cam_data} )

        return disparity_map

    def get_name(self):
        return 'TDNet with VGG11 encoder V2'


class ModelTraining:

    def __init__(self, model, data_files, batch_size = 10, epochs = 20, model_checkpoint_dir = 'model/TD_net.ckpt'):
        self.model = model
        self.train_data_files = data_files # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.batch_size = batch_size
        self.no_epochs = epochs
        self.no_data = len( self.train_data_files )
        self.model_checkpoint_dir = model_checkpoint_dir

    def train_model(self):
        data_points = np.zeros(len(self.train_data_files))
        print ( 'Training Model: %s ... ' % self.model.get_name() )
        # shuffle and read a batch from the train dataset
        # train model for one epoch - call fn model.train_batch(data) for each batch
        for i in range( self.no_epochs ):
            data_shuffled = self.train_data_files[:]
            shuffle(data_shuffled)
            training_loss = 0.0
            for batch in range( 0, self.no_data, self.batch_size ):
                train_data = read_data( data_shuffled[ batch:batch + self.batch_size ] )
                self.model.train_batch( train_data )
                training_loss += self.model.compute_loss( train_data )
                #print("Batch no. : ",batch)
            training_loss /= ( self.no_data // self.batch_size )

            # validate the model and print test, validation accuracy
            batch_idx = next_batch( self.no_data, self.batch_size )
            validation_data = read_data( [ self.train_data_files[ idx ] for idx in batch_idx ] )
            validation_loss = self.model.compute_loss( validation_data )
            valid_output = self.model.forward_pass( validation_data )

            print ( 'epoch: %4d    train loss: %20.4f     val loss: %20.4f' %
                                    ( i, training_loss, validation_loss ) )

            print ('Mean:', np.mean(valid_output))
            print ('Max:', np.max(valid_output))
            print ('Min:', np.min(valid_output))
            print ('Unique:', np.unique(valid_output))
            print ('')
            self.model.save_model( self.model_checkpoint_dir )

        print ( 'Training Model: %s ... Complete' % self.model.get_name() )

    def get_model(self):
        return self.model


class ModelTesting:

    def __init__(self, model, data_files, save_dir = 'test_out'):
        self.model = model
        self.test_data_files = data_files # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.dir_out = save_dir

    def test_model(self):
        directory = self.dir_out + '/' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')
        os.makedirs(directory)

        print ('Testing model:', self.model.get_name() )

        for i, data_file in enumerate(self.test_data_files):
            data = read_data( [ data_file  ] )

            disparity_map = self.model.forward_pass( data )
            test_loss = self.model.compute_loss( data )

            # Save Disparity map
            print ('data',data[2][0])
            print ('disp', disparity_map[0])
            disparity_map = disparity_map[0]
            h, w = disparity_map.shape[0], disparity_map.shape[1]
            disparity_map = disparity_map.reshape((h, w))
            plt.imshow(data[2][0].reshape((h, w)))
            plt.show()
            plt.imshow(disparity_map)
            plt.show()
            fname = self.test_data_files[i][0].split('.')[0] + '_disp.png'
            fname = fname.replace('/', '-')
            fname = directory + '/' + fname
            imsave(fname, disparity_map)

            print ( 'Test: %4d image: %s loss: %4.4f' %( i, self.test_data_files[i][0] , test_loss ) )


    def set_test_data_files(self, files):
        self.test_data_files = files

def main():
    global LEFT_CAM_DIR, RIGHT_CAM_DIR, DISPARITY_DIR

    #net = TDNet_VGG11([480, 640, 3], [480, 640, 1], [60, 80, 256], 'data/vgg16.npy', learning_rate = 1e-4)
    net = TDNet_VGG11_V2([480, 640, 3], [480, 640, 1], 'data/vgg16.npy', learning_rate = 1e-4)
    net.build_model()
    net.add_loss_optimizer()
    net.sess_init()

    # data_files = get_data_files_SINTEL(LEFT_CAM_DIR, RIGHT_CAM_DIR, DISPARITY_DIR)
    data_files = get_data_files_NTSD(CAM_DIR, DISPARITY_DIR)
    data_shuffled = data_files[:]
    shuffle(data_shuffled)

    train_data_files = data_shuffled[len(data_shuffled)//5:]
    test_data_files  = data_shuffled[:len(data_shuffled)//5]

    print ('Test and Train split:', len(train_data_files), len(test_data_files))

    train = ModelTraining(net, train_data_files, batch_size = 5, epochs = 50)
    train.train_model()

if __name__ == '__main__':
    main()
