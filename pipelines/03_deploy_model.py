import mlflow
import shutil
import os
import sys

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config


def prepare_inference_model():
    """
    Fetches the production model from the MLflow Model Registry and saves it locally.
    This script is intended to be run as a deployment step, before starting the
    inference service.
    """
    print("--- Preparing Model for Inference ---")

    # 1. Load Configuration
    print("Loading configuration...")
    config = load_config("config/config.yml")
    mlflow_config = config["mlflow"]
    model_name = mlflow_config["model_name"]
    local_model_dir = "inference_model"

    # 2. Set MLflow Tracking URI
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

    # 3. Define Model URI for the production model
    # The format `models:/<model_name>@<alias>` points to a model alias.
    model_uri = f"models:/{model_name}@prod"
    print(f"Fetching model with alias 'prod' from: {model_uri}")

    # 4. Download the model artifact from MLflow
    try:
        # Remove old model directory if it exists
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir)
            print(f"Removed existing model directory: {local_model_dir}")

        # Download the model artifacts to a local directory
        # This downloads the entire model folder, including the model, conda.yaml, etc.
        model_path = mlflow.artifacts.download_artifacts(
            model_uri, dst_path=local_model_dir
        )
        os.rename(model_path, local_model_dir)

        print(
            f"Successfully downloaded and saved production model to: {local_model_dir}"
        )
        print("The inference service can now load the model from this local directory.")

    except Exception as e:
        print(f"Error fetching model from MLflow Registry: {e}")
        print(
            "Please ensure a model named '{model_name}' has a version in 'Production' stage."
        )
        sys.exit(1)


if __name__ == "__main__":
    prepare_inference_model()
