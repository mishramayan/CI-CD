import os
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
import keras
import random
from keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# MLFlow function
def log_to_mlflow(model, x_train, x_test, y_train, y_test, tflite_model, num_classes, input_shape, batch_size, epochs):
    
    # Set MLflow tracking URI and experiment name from environment variables
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mnist_training_experiment")
    
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
        
    # Create a new experiment if it doesn't exist
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID: {experiment_id}, Name: {experiment_name}")
    
    # Start an MLflow run and set run name from environment variable
    with mlflow.start_run(run_name=os.getenv("MLFLOW_RUN_NAME"), experiment_id=experiment_id) as run:
        # Set user tags
        mlflow.set_tag("build_id", os.getenv("MLFLOW_PIPELINE_BUILD_ID", "local"))
        mlflow.set_tag("run_by", os.getenv("MLFLOW_PIPELINE_RUN_BY", os.getenv("MLFLOW_TRACKING_USERNAME")))
        
        # Log parameters
        mlflow.log_params({
            "num_classes": num_classes,
            "input_shape": input_shape,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer": "adam",
            "loss_function": "categorical_crossentropy"
        })
        
        # train the model and log metrics for each epoch
        log_each_epoch = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
        )
        
        for epoch, loss, acc in zip(range(epochs), log_each_epoch.history['loss'], log_each_epoch.history['accuracy']):
            mlflow.log_metrics({
                "training_loss": loss,
                "training_accuracy": acc
            }, step=epoch)
            
        # Evaluate the model and log evaluation metrics
        score = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metrics({
            "test_loss": score[0],
            "test_accuracy": score[1]
        })
        
        # Log the trained Keras model
        mlflow.keras.log_model(model, "keras_model")
        
        # Quantize the model to TFLite and log as an artifact
        tflite_model_path = "mnist_int8_conv.tflite"
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        mlflow.log_artifact(tflite_model_path)
        
        # Log the .cc files as artifacts in MLflow
        artifact_path = os.path.join(os.getcwd(), "SDK_code")
        mlflow.log_artifacts(artifact_path, artifact_path="SDK_code")
        
        print("Run logged to MLflow: ")
        print(f"Run ID: {run.info.run_id}")

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

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Consistent results for reproducibility
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

# Training parameters
batch_size = 128
epochs = 30
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

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

# Call MLflow logging function
log_to_mlflow(model, x_train, x_test, y_train, y_test, tflite_model, num_classes, input_shape, batch_size, epochs)
