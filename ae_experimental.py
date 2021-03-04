import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt



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




# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


y = CustomVariationalLayer()([x, x_decoded_mean_squash])
vae = Model(x, y)

sgd = optimizers.SGD(lr=0.01)
vae.compile(optimizer=sgd, loss=None)
vae.summary()




def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])







def make_encoder(input_shape):
    encoder_input = keras.Input(shape=input_shape)
    x = layers.Conv2D(3, kernel_size=(2,2), padding='same', activation='relu')(encoder_input)
    x = layers.Conv2D(32, kernel_size=(2,2), padding='same', activation='relu', strides=(2,2))(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', strides=1)(x)
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', strides=1)(x)

    flat_layer = layers.Flatten()(x)
    encoder_output = layers.Dense(32)(flat_layer)

    encoder = keras.Model(encoder_input, encoder_output, name="encoder")
    return encoder


def make_decoder(output_size):
    decoder_input = keras.Input(shape=(32,))
    x = layers.Dense(32)(decoder_input)
    x = layers.Dense(32 * (output_size[0] / 2) * (output_size[1] / 2))(x)

    x = layers.Reshape((int(output_size[0] / 2), int(output_size[1] / 2), 32))(x)

    x = layers.Conv2DTranspose(32, kernel_size=3, padding='same', strides=1, activation='relu')(x)
    x = layers.Conv2DTranspose(32, kernel_size=3, padding='same', strides=1, activation='relu')(x)
    x = layers.Conv2DTranspose(32, kernel_size=(2,2), padding='valid', strides=(2,2), activation='relu')(x)
    decoder_output = layers.Conv2D(3, kernel_size=(2,2), padding='same', activation='sigmoid')(x)

    decoder = keras.Model(decoder_input, decoder_output, name="decoder")
    return decoder



def make_vae(encoder, decoder, input_shape):
    z = encoder(input_shape)
    output = decoder(z)
    vae = keras.Model(input_shape, output, name="vae")
    return vae



#CONFIG
s = (32,32,3)
#s = (28,28,1)
#Select static Sample data ranging [x:y-1]
number_of_pics = 10
sample_data = pic_data[0:number_of_pics]
#sample_data = x_train[0:number_of_pics]
# Number of epochs to run for
max_epochs = 10
num_rows_plot = 5

training_data = pic_data
#training_data = x_train


en = make_encoder(s)
en.summary()
de = make_decoder(s)
de.summary()


vae = make_vae(en, de, keras.Input(shape=s))

vae.summary()

vae.compile(optimizer='adam', loss='binary_crossentropy')






#################################################################







# Create Plotter Function
def plot_step(vae, sample_d, ax, n, plot_i):
    #Function here
    # Plot and display result

    # Simulate Predictions
    # Run encoder and grab variable [2] (Latent data representation)
    # Run decoder on latent space
    result = vae.predict(sample_data)
    for i in range(n):
        ax[plot_i, i].imshow(result[i], cmap=plt.cm.binary)
        ax[plot_i,i].axis("off")
        

    #plt.imshow(result[0], cmap=plt.cm.binary)
    #plt.show()







# Number of Rows to plot
epoch_plot_step = [i for i in range(0,max_epochs,max_epochs // num_rows_plot)]


rows = 4 # defining no. of rows in figure
cols = 12 # defining no. of colums in figure
f = plt.figure(figsize=(2*cols,2*rows)) 
f.tight_layout()




# Setup Plot
# Should have the same number of rows as the sample data length
f, axxar = plt.subplots(num_rows_plot+1, number_of_pics)

for i in range(number_of_pics):
    axxar[0,i].imshow(sample_data[i], cmap=plt.cm.binary)
    axxar[0,i].axis("off")
#plt.show()

#exit()

plot_iter = 1
for epoch in range(max_epochs):
    history = vae.fit(training_data, training_data, epochs=1)
    #f.add_subplot(rows,cols, epoch+1)

    if epoch in epoch_plot_step:
        plot_step(vae, sample_data, axxar, number_of_pics, plot_iter)
        plot_iter += 1

plt.show()



