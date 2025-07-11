import mlflow
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
import time
import os
import sys

# Add project root to Python path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.lstm import LSTMModel
from src.model.wrapper import LSTMWrapper
from src.utils.config_loader import load_config

def run_training():
    """Main function to run the training pipeline."""
    print("--- Starting Training Pipeline ---")

    # 1. Load Configuration
    print("Loading configuration...")
    config = load_config("config/config.yml")
    mlflow_config = config["mlflow"]
    model_params = config["model"]["lstm"]
    data_config = config["data"]
    artifacts_config = config["artifacts"]

    # 2. Setup MLflow
    print(f"Setting up MLflow tracking URI: {mlflow_config['tracking_uri']}")
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    # 3. Load and Prepare Data
    print("Loading and preparing data...")
    df = pd.read_csv(data_config["processed"]["train"])
    
    # Scale the demand feature
    scaler = MinMaxScaler()
    df["demand"] = scaler.fit_transform(df[["demand"]])
    df["demand_diff"] = df["demand"].diff()
    df = df.dropna().reset_index(drop=True)
    
    # Convert to tensor for the model
    features_tensor = torch.tensor(df[["demand", "demand_diff"]].values, dtype=torch.float32)

    # 4. Train the Model
    with mlflow.start_run(run_name=f"{mlflow_config['model_name']}_training") as run:
        print(f"MLflow run started (ID: {run.info.run_id})")
        
        # Log parameters
        mlflow.log_params(model_params)
        print("Logged model parameters to MLflow.")

        # Initialize and train the model
        model = LSTMModel(
            params=model_params,
            input_size=model_params["input_size"],
            output_size=model_params["output_size"]
        )
        
        start_time = time.time()
        trained_model, losses = model.train_model(features_tensor)
        end_time = time.time()
        
        # Log metrics
        mlflow.log_metric("training_time_seconds", end_time - start_time)
        for i, loss in enumerate(losses):
            mlflow.log_metric("train_loss_MSE", loss, step=i)
        print("Logged training metrics to MLflow.")

        # 5. Log Artifacts and Model
        print("Logging scaler and model to MLflow...")
        
        # Log the scaler
        mlflow.sklearn.log_model(
            sk_model=scaler,
            artifact_path=artifacts_config["scaler_path"],
            registered_model_name=f"{mlflow_config['model_name']}_scaler"
        )

        # Log the PyTorch model using the wrapper
        mlflow.pyfunc.log_model(
            artifact_path=artifacts_config["model_path"],
            python_model=LSTMWrapper(trained_model),
            registered_model_name=mlflow_config["model_name"]
        )
        print("Scaler and model logged and registered in MLflow.")

    print("--- Training Pipeline Complete! ---")
    print("Check the MLflow UI to see the results.")

if __name__ == "__main__":
    run_training()
