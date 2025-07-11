import mlflow
import torch
import pandas as pd


class LSTMWrapper(mlflow.pyfunc.PythonModel):
    """
    Wraps the LSTMModel for use with MLflow pyfunc.
    Handles the conversion of pandas DataFrame input to PyTorch Tensors.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Performs inference on the input DataFrame.
        `model_input` is expected to have columns that can be directly converted to a tensor.
        """
        # Convert pandas DataFrame to numpy array, then to a tensor
        input_tensor = torch.tensor(model_input, dtype=torch.float32)

        # Get predictions from the underlying model
        predictions = self.model.predict(input_tensor)

        # Return predictions as a pandas DataFrame
        return pd.DataFrame([predictions], columns=["prediction"])
