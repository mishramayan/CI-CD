import os
import tensorflow as tf
import keras
import numpy as np


# Load the model
model_load_path = "./output/model.keras"
run_id_path = "./output/run_id.txt"

model = tf.keras.models.load_model(model_load_path)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train - 128).astype(np.int8)
x_train = np.expand_dims(x_train, -1)


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

tflite_model_path = "./output/mnist_int8_conv.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)