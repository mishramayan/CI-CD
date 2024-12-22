from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import random

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Convert the images to int8 format from uint8
x_train = (x_train - 128).astype(np.int8)
x_test = (x_test - 128).astype(np.int8)

# Make sure images have shape (a channel dimension in the end)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# consistent results for reproducibility
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


# CNN architecture
model = keras.Sequential([
    keras.Input(shape=input_shape),  # Input: (28, 28, 1) for MNIST
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(num_classes, kernel_size=(3, 3), activation="softmax", padding="valid"),  # Reduce spatial size
    layers.Flatten()  # Output: (None, 10)
])

# Train model
batch_size = 128
epochs = 30
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Print eval
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# quantize to int8 tflite
# representative dataset: pass data in float format with the same preprocessing that was done during training
def representative_dataset():
    for _ in range(100):
      data_sample = x_train[np.random.randint(0, x_train.shape[0], size=1)]
      yield [data_sample.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_model = converter.convert()

int8_model = "mnist_int8_conv.tflite"
with open(int8_model, 'wb') as f:
  f.write(tflite_model)