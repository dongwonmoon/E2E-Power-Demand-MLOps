import mlflow
import yaml
import json
import numpy as np
import pandas as pd
from collections import deque
from kafka import KafkaConsumer
import joblib
import os
import sys

# Add project root for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config


def run_inference():
    """
    Runs a real-time inference loop using a locally stored model,
    consuming data from a Kafka topic.
    """
    print("--- Starting Real-time Inference Service ---")

    # 1. Load Configuration
    print("Loading configuration...")
    config = load_config("config/config.yml")
    model_params = config["model"]["lstm"]
    local_model_dir = "inference_model"  # Standardized local model directory

    # 2. Load the locally prepared model and scaler
    print(f"Loading model from local directory: {local_model_dir}")
    try:
        model = mlflow.pyfunc.load_model(local_model_dir)
        scaler = joblib.load("scaler.pkl")  # Load the scaler saved during processing
    except Exception as e:
        print(f"Error loading local model or scaler: {e}")
        print(
            f"Please run 'python scripts/prepare_inference.py' and ensure 'scaler.pkl' exists."
        )
        sys.exit(1)

    print("Model and scaler loaded successfully.")

    # 3. Initialize data buffer and Kafka Consumer
    seq_len = model_params["seq_len"]
    data_buffer = deque(maxlen=seq_len)

    # --- New Logic: Initialize buffer with the tail of the training data ---
    print("Initializing data buffer with seed data from training set...")
    train_df = pd.read_csv(config["data"]["processed"]["train"])

    # Ensure we don't take more data than available
    num_seed_points = min(len(train_df), seq_len)
    seed_data = train_df["demand"].tail(num_seed_points).tolist()
    data_buffer.extend(seed_data)
    print(f"Buffer initialized with {len(data_buffer)} data points.")
    # --- End of New Logic ---

    consumer = KafkaConsumer(
        "power_demand",
        bootstrap_servers=["localhost:9092"],
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",  # Start from the latest message
    )
    print("🚀 Kafka Consumer is now listening for stream data...")

    # 4. Real-time Inference Loop
    for message in consumer:
        new_data_point = message.value["demand"]
        data_buffer.append(new_data_point)

        if len(data_buffer) == seq_len:
            # Prepare input for the model
            input_df = pd.DataFrame(list(data_buffer), columns=["demand"])

            # Apply the same transformations as in training
            scaled_input = scaler.transform(input_df)
            input_df["demand"] = scaled_input
            input_df["demand_diff"] = input_df["demand"].diff()

            # The model needs the last `seq_len - 1` rows of the diff
            model_input_df = input_df.dropna().reset_index(drop=True)

            # Predict
            prediction_scaled = model.predict(model_input_df.values)

            # Inverse transform the prediction to get the real value
            # The prediction is a DataFrame, so we access the value with .values
            prediction = scaler.inverse_transform(prediction_scaled.values)

            last_actual = input_df["demand"].iloc[-1]
            predicted_value = prediction[0][0]

            print(
                f"📥 Last Actual (Scaled): {last_actual:.4f} -> 📤 Predicted Demand: {predicted_value:.2f}"
            )


if __name__ == "__main__":
    run_inference()
