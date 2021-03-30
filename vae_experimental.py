import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid



#Data Builder File: ./data_builder.py
import data_builder as datab


#MNIST
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


#CIFAR10 Filename List for importer
CIFAR10_Filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']

#Import data from CIFAR10 Dataset. Expected 5000 images total.
# NOTE: This should work properly...
# TODO: Double check the data returned is what is expected
# load_data_sets(file_list, data_id)
# Default data ID is 3 for Cats - See data_builder.py for details
pic_data = datab.load_data_sets(CIFAR10_Filenames)



#CONSTANTS

LATENT_DIM = 128

input_shape = (32,32,3)

# Sampling Function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM),mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon

#Make the encoder here
encoder_input = keras.Input(shape=input_shape)
x = layers.Conv2D(3, kernel_size=(2,2), padding='same', activation='relu')(encoder_input)
x = layers.Conv2D(32, kernel_size=(2,2), padding='same', activation='relu', strides=(2,2))(x)
x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', strides=1)(x)
x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', strides=1)(x)

flat_layer = layers.Flatten()(x)
hidden_layer = layers.Dense(512)(flat_layer)
z_mean = layers.Dense(LATENT_DIM)(hidden_layer)
z_log_var = layers.Dense(LATENT_DIM)(hidden_layer)

encoder_output = layers.Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean, z_log_var])

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()
##########################################################

#Make the decoder Here
decoder_input = keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(512)(decoder_input)
x = layers.Dense(64 * (input_shape[0] / 2) * (input_shape[1] / 2))(x)

x = layers.Reshape((int(input_shape[0] / 2), int(input_shape[1] / 2), 64))(x)

x = layers.Conv2DTranspose(64, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = layers.Conv2DTranspose(64, kernel_size=3, padding='same', strides=1, activation='relu')(x)
x = layers.Conv2DTranspose(32, kernel_size=(2,2), padding='valid', strides=(2,2), activation='relu')(x)
decoder_output = layers.Conv2D(3, kernel_size=(2,2), padding='same', activation='sigmoid')(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()
##########################################################



#Make the VAE here
z = encoder(encoder_input)
output = decoder(z)
vae = keras.Model(encoder_input, output, name="vae")
vae.summary()
#exit()

# Custom Loss Function
# def VAE_loss_function(y_true, y_pred):
base_truth = K.flatten(encoder_input)
predicted_truth = K.flatten(output)
bc_loss = 32 * 32 * keras.losses.binary_crossentropy(base_truth, predicted_truth)
kl_loss = (-0.5) * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
total_loss = K.mean(bc_loss + kl_loss)
##########################################################

vae.add_loss(total_loss)
vae.compile(optimizer='adam')
##########################################################


#Setup training and validation data
training_data = pic_data[:4000]
validation_data = pic_data[4000:]


#CONFIG
#Select static Sample data ranging [x:y-1]
number_of_pics = 10
sample_data = training_data[0:number_of_pics]
sample_data_v = validation_data[0:number_of_pics]

# Number of epochs to run for
max_epochs = 10
num_rows_plot = 5


#################################################################







# Create Plotter Function
def plot_step(vae, target_ims, g, n, plot_i):
    #Function here
    # Plot and display result

    # Simulate Predictions
    # Run encoder and grab variable [2] (Latent data representation)
    # Run decoder on latent space
    result = vae.predict(target_ims)
    offset = n*(plot_i+1)
    g[offset].set_ylabel('EPOCH {}'.format(plot_i*(max_epochs//num_rows_plot)))
    for i in range(n):
        g[offset+1].set_aspect('equal')
        g[offset+i].imshow(result[i], cmap=plt.cm.binary)
        g[offset+i].set_xticklabels([])
        g[offset+i].set_yticklabels([])
        







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
    grid_v[i].imshow(validation_data[i], cmap=plt.cm.binary)
    grid_v[i].set_xticklabels([])
    grid_v[i].set_yticklabels([])


plot_iter = 0
for epoch in range(max_epochs):
    history = vae.fit(training_data, training_data, epochs=1, validation_data=(validation_data, validation_data))

    if epoch in epoch_plot_step:
        plot_step(vae, sample_data, grid, number_of_pics, plot_iter)
        plot_step(vae, validation_data, grid_v, number_of_pics, plot_iter)
        plot_iter += 1


fig.savefig("training-results.png")
fig.show()

fig_v.savefig("validation-results.png")
fig_v.show()





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



