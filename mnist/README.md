# MNIST CI/CD Pipeline Environment Variables

This README describes the environment variables used in the MNIST CI/CD pipeline and instructions for updating them in Azure DevOps pipelines.

---

## **Environment Variables**

The following environment variables are critical for the pipeline's successful execution:

| Variable Name                | Description                                          | Example Value                |
|------------------------------|------------------------------------------------------|------------------------------|
| MLFLOW_TRACKING_URI          | URI of MLflow tracking server.                       | mlflow_url                   |
| MLFLOW_TRACKING_USERNAME     | Username for authenticating with MLflow server.      | mlflow_user                  |
| MLFLOW_TRACKING_PASSWORD     | Password for authenticating with MLflow server.      | your_secure_password         |
| MLFLOW_TRACKING_INSECURE_TLS | To allow insecure TLS connection with MLflow server. | true                         |
| MLFLOW_EXPERIMENT_NAME       | Name of the experiment in MLflow.                    | mnist-training-cnn           |
| MLFLOW_RUN_NAME              | Name of individual run within the experiment.        | mnist-training-run1          |
