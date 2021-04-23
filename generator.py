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
import random


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

decoder = tf.keras.models.load_model('model/VAE_decoder')
encoder = tf.keras.models.load_model('model/VAE_encoder')
#VAE = tf.keras.models.load_model('model/VAE_full')


##########################################

rows = 10
ims_per_row = 10


#Image Plotting Here
total_plot = rows*ims_per_row

fig = plt.figure(figsize=(ims_per_row, rows))
fig.set_size_inches(40,40)
grid = ImageGrid(fig, 111, nrows_ncols=(ims_per_row, rows), axes_pad=0.1)

fig_o = plt.figure(figsize=(ims_per_row, rows))
fig_o.set_size_inches(40,40)
grid_o = ImageGrid(fig_o, 111, nrows_ncols=(ims_per_row, rows), axes_pad=0.1)


##########################################

#Load Manifest
mf_file = open("train.manifest", "r")
data = mf_file.read()
training_manifest = data.split(" ")
mf_file.close()

# Generate new images here


for i in range(total_plot):
	noise = genRandData(512)

	base_im = load_manifest_rand(training_manifest, IMAGE_DIMENSIONS, 1)
	enc_im = encoder.predict(base_im)
	
	#Perturb
	r_dim=random.randint(0,511)
	r_val=random.randint(-100,100)
	enc_im[0][r_dim] = r_val

	results = decoder.predict(enc_im)
	
	grid[i].set_aspect('equal')
	grid[i].imshow(results[0], cmap = plt.cm.binary)
	grid_o[i].set_aspect('equal')
	grid_o[i].imshow(base_im[0], cmap = plt.cm.binary)
	print("Image " + str(i) + " Complete!")
	print("Dim: " + str(r_dim) + " | Val: " + str(r_val))


#plt.show()
fig.savefig("GenImages.png")
fig_o.savefig("GenImages_original.png")
