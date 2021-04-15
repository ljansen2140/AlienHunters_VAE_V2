# def classify_function(images): 

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

#VAE Model File: ./classifier.py
import classifier as cla

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


# Classifier Function 
def classifier_function(images): 

	classifier = tf.keras.models.load_model('CNN_CIFAR.h5')


# Predicting the test data
	predictions = model.predict(########X_test)
	predictions = one_hot_encoder.inverse_transform(predictions)

	
	# Don't think we need this but not sure 
	######y_test = one_hot_encoder.inverse_transform(y_test)


	results = decoder.predict(g1.###### (this should be data - images??) normal(shape=[1,32]))




# Displaying test data with its actual and predicted label - Plotting
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

	y_test = y_test.astype(int)
	predictions = predictions.astype(int)

	fig, axes = plt.subplots(ncols=7, nrows=3, sharex=False,
    	sharey=True, figsize=(17, 8))
	index = 0
	for i in range(3):
    	for j in range(7):
        	axes[i,j].set_title('actual:' + labels[y_test[index][0]] + '\n' 
                            + 'predicted:' + labels[predictions[index][0]])
        	axes[i,j].imshow(X_test[index], cmap='gray')
        	axes[i,j].get_xaxis().set_visible(False)
        	axes[i,j].get_yaxis().set_visible(False)
        	index += 1
	plt.show()






