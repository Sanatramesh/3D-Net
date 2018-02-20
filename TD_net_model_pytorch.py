import os
# import cv2
import sys
import time
import datetime
import argparse
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
from scipy.misc import imread, imsave
from collections import OrderedDict

import torch as th
from torch.autograd import Variable
from torchvision import transforms, utils
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader

# DISP_FILE_TYPE = 'PNG_SINTEL'
# LEFT_CAM_DIR  = 'data/MPI-Sintel-stereo-training-20150305/training/clean_left'
# RIGHT_CAM_DIR = 'data/MPI-Sintel-stereo-training-20150305/training/clean_right'
# DISPARITY_DIR = 'data/MPI-Sintel-stereo-training-20150305/training/disparities'

DISP_FILE_TYPE = 'PNG_NTSD'
CAM_DIR  = 'data/NTSD-200/illumination'
DISPARITY_DIR = 'data/NTSD-200/groundtruth/disparity_maps'
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
    disp_map = np.transpose(disp_map, (2, 0, 1))
    return disp_map

def read_img(img_file):
    img = imread(img_file)
    img = img[:,:,:3]
    img = np.transpose(img, (2, 0, 1))
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

    data[0] = np.array(data[0])
    data[1] = np.array(data[1])
    data[2] = np.array(data[2])

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


def next_batch(no_data_points, batch_size):
    points = np.random.choice( no_data_points, batch_size )
    return list(points)


class TDNet_VGG11_Model(th.nn.Module):

    def __init__(self, freeze_encoder = True):
        super(TDNet_VGG11_Model, self).__init__()
        self.vgg_encoder = th.nn.Sequential(
                            th.nn.Conv2d(3, 64, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.Conv2d(64, 64, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.MaxPool2d(2, stride = 2),

                            th.nn.Conv2d(64, 128, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.Conv2d(128, 128, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.MaxPool2d(2, stride = 2),

                            th.nn.Conv2d(128, 256, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.Conv2d(256, 256, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.MaxPool2d(2, stride = 2)
                    )

        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
        self.vgg_encoder.load_state_dict(state_dict, strict = False)

        if freeze_encoder:
            for param in self.vgg_encoder.parameters():
                param.requires_grad = False

        self.decoder = th.nn.Sequential(
                        th.nn.Conv2d(256, 256, 3, padding = 1),
                        th.nn.ReLU(),
                        th.nn.ConvTranspose2d(256, 256, 4, stride = 2, padding = 1),

                        th.nn.Conv2d(256, 128, 3, padding = 1),
                        th.nn.ReLU(),
                        th.nn.ConvTranspose2d(128, 128, 4, stride = 2, padding = 1),

                        th.nn.Conv2d(128, 64, 3, padding = 1),
                        th.nn.ReLU(),
                        th.nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),

                        th.nn.Conv2d(64, 1, 3, padding = 1),
                        th.nn.ReLU()
                    )

    def forward(self, left_img, right_img):
        left_feat = self.vgg_encoder(left_img)
        right_feat = self.vgg_encoder(right_img)

        # feature combine
        # features = th.mul(left_feat, right_feat)
        features = th.add(left_feat, right_feat)
        # features = th.cat(( left_feat, right_feat ), 3)

        y = self.decoder(features)

        return y

    def get_name(self):
        return 'TDNet with VGG11 encoder'

class TDNet(object):

    def __init__(self, input_dims, output_dims, learning_rate = 1e-4):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate

    def build_model(self):
        self.model = TDNet_VGG11_Model()
        print (self.model)
    def loss_func(self, y, y_):
        loss = th.mean((y - y_) ** 2)
        return loss

    def add_optimizer(self):
        self.optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.learning_rate)

    def train_batch(self, data):
        left_cam_data, right_cam_data, disp_data = data
        left_cam_data, right_cam_data, disp_data = th.autograd.Variable(th.FloatTensor(left_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(right_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(disp_data))

        y = self.model.forward( left_cam_data, right_cam_data )
        print (y.shape, disp_data.shape)
        loss = self.loss_func( y, disp_data )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def compute_loss(self, data):
        left_cam_data, right_cam_data, disp_data = data
        left_cam_data, right_cam_data, disp_data = th.autograd.Variable(th.FloatTensor(left_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(right_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(disp_data))

        y = self.model.forward( left_cam_data, right_cam_data )
        loss = self.loss_func( y, disp_data )

        return loss.data

    def forward_pass(self, data):
        left_cam_data, right_cam_data, disp_data = data
        left_cam_data, right_cam_data, disp_data = th.autograd.Variable(th.FloatTensor(left_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(right_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(disp_data))

        y = self.model.forward( left_cam_data, right_cam_data )
        disparity_map = y.data

        return disparity_map

    def save_model(self, ckpt_file):
        # th.save( self.model, ckpt_file )
        self.model.save_state_dict(ckpt_file)

    def load_model(self, ckpt_file = None):
        if ckpt_file != None:
            self.model.load_state_dict(th.load(ckpt_file))

    def get_name(self):
        return self.model.get_name()


class ModelTraining:

    def __init__(self, model, data_files, batch_size = 10, epochs = 20, model_checkpoint_dir = 'model/TD_net.pt'):
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
                print (batch,)
                train_data = read_data( data_shuffled[ batch:batch + self.batch_size ] )
                training_loss += self.model.train_batch( train_data )

            training_loss /= ( self.no_data // self.batch_size )
            print ()

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

    net = TDNet([480, 640, 3], [480, 640, 1], learning_rate = 1e-4)
    net.build_model()
    net.add_optimizer()

    # data_files = get_data_files_SINTEL(LEFT_CAM_DIR, RIGHT_CAM_DIR, DISPARITY_DIR)
    data_files = get_data_files_NTSD(CAM_DIR, DISPARITY_DIR)
    data_shuffled = data_files[:]
    shuffle(data_shuffled)

    train_data_files = data_shuffled[len(data_shuffled)//5:]
    test_data_files  = data_shuffled[:len(data_shuffled)//5]

    print ('Test and Train split:', len(train_data_files), len(test_data_files))

    train = ModelTraining(net, train_data_files, batch_size = 1, epochs = 10)
    train.train_model()

if __name__ == '__main__':
    main()
