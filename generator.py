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


# Random Data Generator
# Returns a random normal distribution of a specified size
def genRandData(size):
	g1 = tf.random.get_global_generator()
	o = g1.normal(shape=[1,size])
	return o



# - Data Needed for Loading the VAE
#CIFAR10 Filename List for importer
CIFAR10_Filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
LATENT_DIM = 512
HIDDEN_LAYER_DIM = 2048

IMAGE_DIMENSIONS = (512,512)

input_shape = IMAGE_DIMENSIONS + (3,)

#CIFAR 10 Data
#pic_data = datab.load_data_sets(CIFAR10_Filenames)


# Load the model parts

decoder = tf.keras.models.load_model('model/VAE_decoder.h5')
encoder = tf.keras.models.load_model('model/VAE_encoder.h5')
VAE = tf.keras.models.load_model('model/VAE.h5')


##########################################

# Generate new images here

new_im = genRandData(512)
results = decoder.predict(new_im)


plt.imshow(results[0], cmap = plt.cm.binary)
#plt.show()
plt.savefig("GenImage.png")
