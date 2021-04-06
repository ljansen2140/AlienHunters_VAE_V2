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

# pic_data = datab.load_data_sets(CIFAR10_Filenames)

# ^^^ Used for CIFAR10

################################################################
#Get Args
architecture_only = False
reload_previous = False
start_at_epoch = 0
if (len(sys.argv) > 1):
    if (sys.argv[1] == "-a" or sys.argv[1] == "--arch"):
        architecture_only = True
        print("Only Displaying Architecture - Model will not be run!")
    if (sys.argv[1] == "-l" or sys.argv[1] == "--load"):
        reload_previous = True
        start_at_epoch = int(sys.argv[2])
        print("Restarting Model at Epoch: " + str(start_at_epoch))

################################################################

#CONSTANTS

LATENT_DIM = 1024
#HIDDEN_LAYER_DIM = 128

IMAGE_DIMENSIONS = (512,512)

input_shape = IMAGE_DIMENSIONS + (3,)


################################################################

# Sampling Function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM),mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon

#Make the encoder here
encoder_input = keras.Input(shape=input_shape)
#Convolutional Layer to make sure 3-channel RGB is represented
x = layers.Conv2D(3, kernel_size=(4,4), padding='same', activation='relu', name='RGB_Layer')(encoder_input)

#Convolutional Layer 1
x = layers.Conv2D(32, kernel_size=(4,4), padding='same', activation='relu', strides=(2,2), name='Conv_Layer_1')(x)
x = layers.MaxPooling2D((2,2), padding='same', name='Pooling_Layer_1')(x)

#Convolutional Layer 2
x = layers.Conv2D(32, kernel_size=(4,4), padding='same', activation='relu', strides=(4,4), name='Conv_Layer_2')(x)
x = layers.MaxPooling2D((4,4), padding='same', name='Pooling_Layer_2')(x)

#Convolutional Layer 3
x = layers.Conv2D(64, kernel_size=(4,4), padding='same', activation='relu', strides=1, name='Conv_Layer_3')(x)
x = layers.MaxPooling2D((4,4), padding='same', name='Pooling_Layer_3')(x)

#Convolutional Layer 4
#x = layers.Conv2D(64, kernel_size=(4,4), padding='same', activation='relu', strides=(4,4), name='Conv_Layer_4')(x)

#Convolutional Layer 5
#x = layers.Conv2D(64, kernel_size=(4,4), padding='same', activation='relu', strides=(2,2), name='Conv_Layer_5')(x)

#Flatten Data and Hidden Layer
flat_layer = layers.Flatten(name='Flatten_Layer')(x)
#hidden_layer = layers.Dense(HIDDEN_LAYER_DIM, name='Hidden_Layer')(flat_layer)

#Latent Space is Built Here
z_mean = layers.Dense(LATENT_DIM, name='Z_MEAN')(flat_layer)
z_log_var = layers.Dense(LATENT_DIM, name='Z_LOG_VAR')(flat_layer)

encoder_output = layers.Lambda(sampling, output_shape=(LATENT_DIM,), name='Latent_Space')([z_mean, z_log_var])

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()


##########################################################

#Make the decoder Here
decoder_input = keras.Input(shape=(LATENT_DIM,))
#Reverse Hidden Layers
#x = layers.Dense(HIDDEN_LAYER_DIM, name='Hidden_Layer')(decoder_input)
x = layers.Dense(64 * 2 * 2, name='Upscale_Layer')(decoder_input)

#Reshape for Conv Layers
x = layers.Reshape((2, 2, 64))(x)

#x = layers.Conv2DTranspose(64, kernel_size=(4,4), padding='same', strides=(2,2), activation='relu', name='TP_Layer_5')(x)

#x = layers.Conv2DTranspose(64, kernel_size=(4,4), padding='same', strides=(4,4), activation='relu', name='TP_Layer_4')(x)

#Convolutional Layers Transpose and UpSampling
x = layers.UpSampling2D((4,4), name="UpSample_Layer_3")(x)
x = layers.Conv2DTranspose(64, kernel_size=(4,4), padding='same', strides=1, activation='relu', name='TP_Layer_3')(x)

x = layers.UpSampling2D((4,4), name="UpSample_Layer_2")(x)
x = layers.Conv2DTranspose(32, kernel_size=(4,4), padding='same', strides=(4,4), activation='relu', name='TP_Layer_2')(x)

x = layers.UpSampling2D((2,2), name="UpSample_Layer_1")(x)
x = layers.Conv2DTranspose(32, kernel_size=(4,4), padding='same', strides=(2,2), activation='relu', name='Transpose_Layer_1')(x)


decoder_output = layers.Conv2D(3, kernel_size=(4,4), padding='same', activation='sigmoid', name='Transpose_RGB_Layer')(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()
##########################################################



#Make the VAE here
z = encoder(encoder_input)
output = decoder(z)


vae = keras.Model(encoder_input, output, name="vae")
vae.summary()

if architecture_only:
    exit()

# Custom Loss Function
# def VAE_loss_function(y_true, y_pred):
base_truth = K.flatten(encoder_input)
predicted_truth = K.flatten(output)
bc_loss = 32 * 32 * keras.losses.binary_crossentropy(base_truth, predicted_truth)
kl_loss = (-0.5) * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
total_loss = K.mean(bc_loss + kl_loss)


vae.add_loss(total_loss)
vae.compile(optimizer='adam')
##########################################################


#Setup training and validation data
#CIFAR10 DATA
# training_data = pic_data[:4000]
# validation_data = pic_data[4000:]


# train_count = 32
# val_count = 8


#Load Manifest
mf_file = open("train.manifest", "r")
data = mf_file.read()
training_manifest = data.split(" ")
mf_file.close()

#Load Validation Manifest
mf_file = open("val.manifest", "r")
data = mf_file.read()
validation_manifest = data.split(" ")
mf_file.close()


### Load Data From Static Location
# print("Loading Training Data...")
# training_data = load_manifest(training_manifest, IMAGE_DIMENSIONS)
# print("Loading Validation Data...")
# validation_data = load_manifest(validation_manifest, IMAGE_DIMENSIONS)



###
#This is all unneccessary now... V
# Only useful for OTF loading

#Split Manifest into Train and Val
# val_target = "val/airport/"
# train_target = "train/airport/"

# train_manifest = []
# val_manifest = []

#Add all targets to respective manifests
# for item in objects:
#     if val_target in item:
#         val_manifest.append(item)
#     if train_target in item:
#         train_manifest.append(item)

#train_manifest: Contains all items that can be drawn as training data
#val_manifest: Contains all items that can be drawn as validation data
#train_count: Number of training items to load
#val_count: Number of validation items to load


################################################################


#CONFIG-VARIABLES
#Select static Sample data ranging [x:y-1]
number_of_pics = 10
# sample_data = training_data[0:number_of_pics]
# sample_data_v = validation_data[0:number_of_pics]

# sample_data = load_im(train_manifest, number_of_pics, IMAGE_DIMENSIONS)
# sample_data_v = load_im(val_manifest, number_of_pics, IMAGE_DIMENSIONS)

sample_data = load_manifest_rand(training_manifest, IMAGE_DIMENSIONS, 10)
sample_data_v = load_manifest_rand(validation_manifest, IMAGE_DIMENSIONS, 10)


# Number of epochs to run for
max_epochs = 100
num_rows_plot = 20

#################################################################




################################################################
#PLOTTING FUNCTIONS

# Create Plotter Function
def plot_step(plot_data, g, n, plot_i):
    #Function here
    # Plot and display result

    # Simulate Predictions
    # Run encoder and grab variable [2] (Latent data representation)
    # Run decoder on latent space
    result = plot_data
    offset = n*(plot_i+1)
    g[offset].set_ylabel('EPOCH {}'.format(plot_i*(max_epochs//num_rows_plot)))
    for i in range(n):
        g[offset+1].set_aspect('equal')
        g[offset+i].imshow(result[i], cmap=plt.cm.binary)
        g[offset+i].set_xticklabels([])
        g[offset+i].set_yticklabels([])
        

#Returns only the numpy array representing the predicted image, should be reconstructed at the end
def gen_sample(vae, input_ims):
    result = vae.predict(input_ims)
    return result


################################################################
#PLOTTING CONFIGURATION

# Number of Rows to plot
epoch_plot_step = [i for i in range(0,max_epochs,max_epochs // num_rows_plot)]


# Setup Plot for Training Images
# Should have the same number of rows as the sample data length
fig = plt.figure(figsize=(number_of_pics, num_rows_plot+1))
fig.set_size_inches(40,40)
grid = ImageGrid(fig, 111, nrows_ncols=(num_rows_plot+1, number_of_pics), axes_pad=0.1)

grid[0].set_ylabel('BASE TRUTH')
for i in range(number_of_pics):
    grid[i].set_aspect('equal')
    grid[i].imshow(sample_data[i], cmap=plt.cm.binary)
    grid[i].set_xticklabels([])
    grid[i].set_yticklabels([])


# Setup Plot for Validation Images
# Should have the same number of rows as the sample data length
fig_v = plt.figure(figsize=(number_of_pics, num_rows_plot+1))
fig_v.set_size_inches(40,40)
grid_v = ImageGrid(fig_v, 111, nrows_ncols=(num_rows_plot+1, number_of_pics), axes_pad=0.1)

grid_v[0].set_ylabel('BASE TRUTH')
for i in range(number_of_pics):
    grid_v[i].set_aspect('equal')
    grid_v[i].imshow(sample_data_v[i], cmap=plt.cm.binary)
    grid_v[i].set_xticklabels([])
    grid_v[i].set_yticklabels([])


################################################################

#Setup Checkpoint Callbacks
checkpoint_path = "model/model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


#If plot has been checkpointed, load it here


#TRAINING HAPPENS HERE


plot_iter = 0
plot_data = np.empty((0,number_of_pics) + IMAGE_DIMENSIONS + (3,))
plot_data_v = np.empty((0,number_of_pics) + IMAGE_DIMENSIONS + (3,))
plot_reshape = (-1,number_of_pics) + IMAGE_DIMENSIONS + (3,)

#reload old plots if asked
if reload_previous:
    plot_data = np.load("training_plot_data_checkpoint.npy")
    plot_data_v = np.load("validation_plot_data_checkpoint.npy")

if reload_previous:
    #Reload Model From Checkpoint
    vae.load_weights(checkpoint_path)
    print("Loaded Last Checkpoint for VAE")

#start_at_epoch, defaults to 0, will start at a later epoch if specified by command line args
for epoch in range(start_at_epoch, max_epochs):

    #!!!
    #TODO: Add asynchronous behavior?
    #NOTE: Current load times of 48 images is ~19.5 seconds 

    #TODO: Remove Timing functions?
    #start_load = time.time()
    #Load data for each epoch, 32 training images, 8 validation images
    #training_data = load_im(train_manifest, 32, IMAGE_DIMENSIONS)
    #validation_data = load_im(val_manifest, 8, IMAGE_DIMENSIONS)
    training_data = load_manifest_rand(training_manifest, IMAGE_DIMENSIONS, 32)
    validation_data = load_manifest_rand(validation_manifest, IMAGE_DIMENSIONS, 8)
    #print("Loaded batch for epoch " + str(epoch) + " in " + str(time.time()-start_load) + " seconds.")
    print("Running Epoch: " + str(epoch))
    history = vae.fit(training_data, training_data, epochs=1, validation_data=(validation_data, validation_data), callbacks=[cp_callback])

    if epoch in epoch_plot_step:
        #plot_step(vae, sample_data, grid, number_of_pics, plot_iter)
        #plot_step(vae, sample_data_v, grid_v, number_of_pics, plot_iter)
        sp = gen_sample(vae, sample_data)
        sp = sp.reshape(plot_reshape)
        plot_data = np.concatenate((plot_data, sp))

        sp = gen_sample(vae, sample_data_v)
        sp = sp.reshape(plot_reshape)
        plot_data_v = np.concatenate((plot_data_v, sp))
        
        plot_iter += 1
        print("Epoch " + str(epoch) + " Plotted.")
        #Save Numpy Data Here
        print("-----Saving Plot Data-----")
        np.save("training_plot_data_checkpoint.npy", plot_data)
        np.save("validation_plot_data_checkpoint.npy", plot_data_v)
        


################################################################
#GENERATE IMAGES HERE
i = 0
for im_row in plot_data:
    plot_step(im_row, grid, number_of_pics, i)
    i += 1

i = 0
for im_row_v in plot_data_v:
    plot_step(im_row_v, grid_v, number_of_pics, i)
    i += 1


################################################################
#SAVE IMAGE RESULTS

fig.savefig("training-results.png")
fig.show()

fig_v.savefig("validation-results.png")
fig_v.show()


################################################################
#STATISTICS

#FIXME: Stats not working???
exit()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()

plt.savefig("loss.png")




plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#plt.show()

plt.savefig("acc.png")