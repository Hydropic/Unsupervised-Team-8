import numpy as np

data = np.load("data_squareRoom.npy")
print(data.shape)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, Model, Input

class ConvLSTMAutoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(ConvLSTMAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.ConvLSTM2D(8, (3, 3), activation="relu", padding="same", return_sequences=True, input_shape=shape),
            layers.MaxPooling3D((1, 2, 2), padding="same"),
            layers.ConvLSTM2D(16, (3, 3), activation="relu", padding="same", return_sequences=False),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(np.prod((shape[0], shape[1]//2, shape[2]//2, 4)), activation='relu'),
            layers.Reshape((shape[0], shape[1]//2, shape[2]//2, 4)),
            layers.UpSampling3D((1, 2, 2)),
            layers.Conv3DTranspose(8, (3, 3, 3), activation='relu', padding="same"),
            layers.Conv3DTranspose(shape[3], (3, 3, 3), activation='sigmoid', padding="same")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Assuming x_train and x_test are your datasets with shape (num_samples, time_steps, height, width, channels)  # Example data with original dimensions
data_subset = data[:len(data)//3]  # Take only a tenth of the data
data_subset = np.expand_dims(data_subset, axis=1)
shape = data_subset.shape[1:]  # (time_steps, height, width, channels)
latent_dim = 3
autoencoder = ConvLSTMAutoencoder(latent_dim, shape)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError(), metrics=["accuracy"])

# Print model summary
autoencoder.summary()
autoencoder.build(input_shape=(None, *shape))
# Fit the model
autoencoder.load_weights("autoencoder.keras")

from tensorflow.keras import layers, losses, Model, Input
# Create a model to get the encoder's output
input_layer = Input(shape=shape)
encoded_output = autoencoder.encoder(input_layer)
encoder_model = Model(inputs=input_layer, outputs=encoded_output)

import matplotlib.pyplot as plt
all_encode = encoder_model.predict(data_subset)
zoom = all_encode[:500]
gradient = np.linspace(0, 1, zoom.shape[0])
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.scatter(zoom[:,0],zoom[:,1],zoom[:,2], cmap="viridis", c=gradient)
plt.show()