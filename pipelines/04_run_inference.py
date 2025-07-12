import mlflow
import json
import pandas as pd
from collections import deque
from kafka import KafkaConsumer
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
    config = load_config("config/config.yml")
    inference_config = config['inference']
    deploy_config = config['deployment']
    model_params = config['training']['model_params']
    local_model_dir = deploy_config['local_model_dir']

    # 2. Load the locally prepared model
    print(f"Loading model from local directory: {local_model_dir}")
    try:
        # The loaded model is the LSTMWrapper, which includes the processor
        model = mlflow.pyfunc.load_model(local_model_dir)
    except Exception as e:
        print(f"Error loading local model: {e}")
        print(f"Please run 'pipelines/03_deploy_model.py' first.")
        sys.exit(1)

    print("Model loaded successfully.")

    # 3. Initialize data buffer and Kafka Consumer
    seq_len = model_params["seq_len"]
    data_buffer = deque(maxlen=seq_len)

    # Initialize buffer with the tail of the training data for a warm start
    print("Initializing data buffer with seed data...")
    seed_df = pd.read_csv(inference_config["buffer_seed_data_path"])
    seed_data = seed_df["demand"].tail(seq_len).tolist()
    data_buffer.extend(seed_data)
    print(f"Buffer initialized with {len(data_buffer)} data points.")

    consumer = KafkaConsumer(
        inference_config['kafka']['topic_name'],
        bootstrap_servers=[inference_config['kafka']['bootstrap_servers']],
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )
    print("🚀 Kafka Consumer is now listening for stream data...")

    # 4. Real-time Inference Loop
    for message in consumer:
        new_data_point = message.value["demand"]
        data_buffer.append(new_data_point)

        if len(data_buffer) == seq_len:
            # Prepare input for the model
            input_df = pd.DataFrame(list(data_buffer), columns=["demand"])

            # Predict using the wrapper (handles all processing)
            prediction_df = model.predict(input_df)
            predicted_value = prediction_df.iloc[0][0]

            last_actual = input_df["demand"].iloc[-1]
            print(
                f"📥 Last Actual: {last_actual:.2f} -> 📤 Predicted Demand: {predicted_value:.2f}"
            )

if __name__ == "__main__":
    run_inference()