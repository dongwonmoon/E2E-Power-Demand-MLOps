import mlflow
import pandas as pd
import torch
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.lstm import LSTMModel
from src.model.wrapper import LSTMWrapper
from src.utils.config_loader import load_config
from src.data.processor import DataProcessor


def run_training():
    """Main function to run the training pipeline."""
    print("--- Starting Training Pipeline ---")

    # 1. Load Configuration
    config = load_config("config/config.yml")
    train_config = config["training"]
    data_config = config["data_processing"]

    # 2. Setup MLflow
    mlflow.set_tracking_uri(train_config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(train_config["mlflow"]["experiment_name"])

    # 3. Load and Prepare Data
    train_data_path = os.path.join(
        data_config["processed_data_dir"], data_config["train_filename"]
    )
    df_train = pd.read_csv(train_data_path)

    processor = DataProcessor(config)
    processor.fit_scaler(df_train)  # Fit scaler on training data
    processed_df = processor.transform(df_train.copy())

    # Save the fitted scaler for inference use
    scaler_path = "scaler/scaler.pkl"
    processor.save_scaler(scaler_path)
    print(f"Scaler has been fitted and saved to {scaler_path}")

    features_tensor = torch.tensor(
        processed_df[["demand_scaled", "demand_diff"]].values, dtype=torch.float32
    )

    # 4. Train the Model
    with mlflow.start_run(
        run_name=f"{train_config['mlflow']['model_name']}_training"
    ) as run:
        print(f"MLflow run started (ID: {run.info.run_id})")

        # Log parameters
        mlflow.log_params(train_config["model_params"])
        mlflow.log_params(train_config["train_params"])
        print("Logged model and training parameters to MLflow.")

        # Initialize and train the model
        model = LSTMModel(
            params={**train_config["model_params"], **train_config["train_params"]},
            input_size=train_config["model_params"]["input_size"],
            output_size=train_config["model_params"]["output_size"],
        )

        trained_model, losses = model.train_model(features_tensor)

        # Log metrics
        for i, loss in enumerate(losses):
            mlflow.log_metric("train_loss_MSE", loss, step=i)
        print("Logged training metrics to MLflow.")

        # 5. Log Artifacts and Model
        print("Logging scaler and model to MLflow...")

        # Log the scaler as an artifact
        mlflow.log_artifact(
            scaler_path, artifact_path=train_config["artifacts"]["scaler_path"]
        )

        # Log the PyTorch model using the wrapper
        mlflow.pyfunc.log_model(
            artifact_path=train_config["artifacts"]["model_path"],
            python_model=LSTMWrapper(
                trained_model, processor
            ),  # Pass processor to wrapper
            registered_model_name=train_config["mlflow"]["model_name"],
        )
        print("Scaler and model logged and registered in MLflow.")

    print("--- Training Pipeline Complete! ---")


if __name__ == "__main__":
    run_training()
