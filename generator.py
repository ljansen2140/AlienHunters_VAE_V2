import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import sys
import pickle
import os

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid



#Data Builder File: ./data_builder.py
import data_builder as datab

#Data Pipeline for FMOW dataset: ./pipeline.py
from pipeline import load_im, load_manifest, load_manifest_count, load_manifest_rand


#MNIST
from keras.datasets import mnist
import numpy as np
# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.


#CIFAR10 Filename List for importer
CIFAR10_Filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']

#Import data from CIFAR10 Dataset. Expected 5000 images total.
# NOTE: This should work properly...
# TODO: Double check the data returned is what is expected
# load_data_sets(file_list, data_id)
# Default data ID is 3 for Cats - See data_builder.py for details

LATENT_DIM = 32
HIDDEN_LAYER_DIM = 2048

IMAGE_DIMENSIONS = (32,32)

input_shape = IMAGE_DIMENSIONS + (3,)

pic_data = datab.load_data_sets(CIFAR10_Filenames)

g1=tf.random.get_global_generator()
print(g1.normal(shape=[1,32]))

decoder = tf.keras.models.load_model('model/VAE_decoder.h5')
VAE = tf.keras.models.load_model('model/VAE.h5')

results = decoder.predict(g1.normal(shape=[1,32]))

plt.imshow(results[0], cmap = plt.cm.binary)
plt.show()
