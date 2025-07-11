# E2E Power Demand MLOps

This project implements an end-to-end MLOps pipeline for power demand forecasting using an LSTM model. It covers data processing, model training with MLflow tracking, model deployment, and real-time inference via Kafka.

## Project Structure

```
E2E-Power-Demand-MLOps/
├── config/             # Configuration files
├── data/               # Raw and processed data
├── mlartifacts/        # MLflow artifacts (local storage)
├── mlruns/             # MLflow runs (local storage)
├── notebooks/          # Jupyter notebooks for exploration and initial development
├── pipelines/          # Core MLOps pipeline scripts (ordered execution)
│   ├── 01_process_data.py
│   ├── 02_train_model.py
│   ├── 03_deploy_model.py
│   └── 04_run_inference.py
├── streaming/          # Scripts for Kafka-based real-time data streaming
│   └── producer.py
├── src/                # Source code for modules
│   ├── consumer/       # Kafka consumer logic
│   ├── data/           # Data processing utilities
│   ├── model/          # LSTM model definition and wrapper
│   └── utils/          # General utilities (config loader, MLflow loader)
├── scaler.pkl          # Saved MinMaxScaler (used by inference)
├── .gitignore
├── docker-compose.yml  # Docker Compose for services like Kafka, MLflow
└── README.md
```

## Features

*   **Data Ingestion & Preprocessing:** Combines raw power demand data, cleans it, and splits it into training and streaming datasets.
*   **LSTM Model Training:** Trains a Long Short-Term Memory (LSTM) neural network for time series forecasting.
*   **MLflow Integration:**
    *   Tracks experiments (parameters, metrics, artifacts).
    *   Manages model versions in the MLflow Model Registry.
    *   Facilitates model deployment by promoting models to 'Production' (via Aliases).
*   **Real-time Inference:** Consumes streaming data from Kafka, performs predictions using the deployed model, and outputs results.
*   **Scalability:** Designed with modular components for potential scaling with Docker/Kubernetes and orchestration tools.

## Technologies Used

*   **Python**
*   **PyTorch:** Deep learning framework for LSTM model.
*   **MLflow:** For MLOps lifecycle management (Experiment Tracking, Model Registry).
*   **Apache Kafka:** For real-time data streaming.
*   **scikit-learn:** For data scaling (MinMaxScaler).
*   **pandas, numpy:** For data manipulation.
*   **YAML:** For configuration management.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Docker (for Kafka and MLflow services)
*   `pip` for package management

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd E2E-Power-Demand-MLOps
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt # You might need to create this file based on your environment
    ```
    *(Note: A `requirements.txt` file is not provided in the initial context. You may need to generate one using `pip freeze > requirements.txt` after installing necessary libraries like `torch`, `mlflow`, `pandas`, `scikit-learn`, `kafka-python`, `pyyaml`, `tqdm`.)*

4.  **Start Kafka and MLflow services using Docker Compose:**
    ```bash
    docker-compose up -d
    ```
    This will start Kafka brokers and an MLflow tracking server (accessible at `http://localhost:5000`).

## Running the MLOps Pipeline

Execute the scripts in the `pipelines/` directory in the following order:

1.  **Data Processing:**
    Combines raw data, preprocesses it, and splits it into training and streaming datasets.
    ```bash
    python pipelines/01_process_data.py
    ```

2.  **Model Training:**
    Trains the LSTM model and logs experiments, metrics, and the model to MLflow.
    ```bash
    python pipelines/02_train_model.py
    ```

3.  **Model Deployment (Prepare for Inference):**
    After training, go to the MLflow UI (`http://localhost:5000`), navigate to the `lstm_power_demand` model, select the desired version, and add the `prod` alias to it. Then, run this script to download the 'prod' aliased model locally for fast inference.
    ```bash
    python pipelines/03_deploy_model.py
    ```

4.  **Start Kafka Producer (in a separate terminal):**
    This script simulates real-time data streaming to Kafka.
    ```bash
    python streaming/producer.py
    ```

5.  **Run Real-time Inference (in another separate terminal):**
    Consumes data from Kafka and performs predictions using the locally deployed model.
    ```bash
    python pipelines/04_run_inference.py
    ```

## TODOs & Future Enhancements

*   **Streamlit Dashboard:** Develop a Streamlit application for visualizing real-time predictions and model performance.
*   **Apache Airflow Integration:** Orchestrate the entire MLOps pipeline (data processing, training, deployment) using Airflow for automated workflows.
*   **Model Monitoring:** Implement monitoring for model drift and performance degradation.
*   **CI/CD Pipeline:** Set up Continuous Integration/Continuous Deployment for automated testing and deployment.
*   **Hyperparameter Tuning:** Integrate hyperparameter optimization (e.g., with Optuna or Hyperopt).
*   **Error Handling & Logging:** Enhance robust error handling and comprehensive logging across all components.
