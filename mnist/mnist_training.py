import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import mlflow
import mlflow.keras


def prepare_data():
    """
    Loads and preprocesses the MNIST dataset.
    Returns preprocessed training and testing datasets along with their labels.
    """
    num_classes = 10
    input_shape = (28, 28, 1)

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Convert the images to int8 format from uint8 and normalize
    x_train = (x_train - 128).astype(np.int8)
    x_test = (x_test - 128).astype(np.int8)

    # Add a channel dimension for compatibility with Conv2D
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, num_classes, input_shape


def build_model(input_shape, num_classes):
    """
    Builds and returns a CNN model for MNIST digit classification.
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),  # Input layer
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(num_classes, kernel_size=(3, 3), activation="softmax", padding="valid"),
        layers.Flatten(),  # Flatten output to match the number of classes
    ])
    return model


def train_and_log(model, x_train, x_test, y_train, y_test, num_classes, input_shape, batch_size, epochs):
    """
    Logs the training process, model, and metrics to MLflow.
    """
    
    # Set up MLflow experiment
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mnist_training_experiment")
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    # Set model name to be logged in the mlflow
    model_name = "mnist_model"

    # Create a new experiment if it doesn't exist
    experiment_id = experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID: {experiment_id}, Name: {experiment_name}")

    # Start MLflow run
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

        # Train the model and log metrics
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
        )
        for epoch, (loss, acc) in enumerate(zip(history.history['loss'], history.history['accuracy'])):
            mlflow.log_metrics({"training_loss": loss, "training_accuracy": acc}, step=epoch)

        # Evaluate the model
        score = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metrics({"test_loss": score[0], "test_accuracy": score[1]})

        # Log the trained Keras model
        mlflow.keras.log_model(model, "keras_model")

        # Save the model to a shared directory
        model_dir = "./output/"
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, "model.keras")
        model.save(model_save_path)
        
        #Save run id to a file for use in the pipeline
        run_id_path = os.path.join(model_dir, "run_id.txt")
        with open(run_id_path, "w") as f:
            f.write(run.info.run_id)

        print(f"Run logged to MLflow with Run ID: {run.info.run_id}")


if __name__ == "__main__":
    # Reproducibility
    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # Prepare data
    x_train, y_train, x_test, y_test, num_classes, input_shape = prepare_data()

    # Build model
    model = build_model(input_shape, num_classes)

    # Compile model
    batch_size = 128
    epochs = 1
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Log to MLflow
    train_and_log(model, x_train, x_test, y_train, y_test, num_classes, input_shape, batch_size, epochs)
