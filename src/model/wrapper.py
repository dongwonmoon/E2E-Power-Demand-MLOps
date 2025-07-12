import mlflow
import torch
import pandas as pd

class LSTMWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Accepts a raw pandas DataFrame with a 'demand' column, processes it,
        makes a prediction, and inverse transforms the result.
        """
        # 1. Process the input data using the stored processor
        processed_input = self.processor.transform(model_input.copy())
        
        # Ensure there's enough data to form a sequence
        if processed_input.empty:
            return pd.DataFrame([float('nan')], columns=["prediction"])

        # 2. Convert to tensor
        input_tensor = torch.tensor(processed_input[['demand_scaled', 'demand_diff']].values, dtype=torch.float32)

        # 3. Predict using the underlying model
        # The model's predict method returns a list or a single value
        predictions_scaled = self.model.predict(input_tensor)

        # 4. Inverse transform the prediction
        # The processor's inverse_transform expects a 2D array-like object
        if isinstance(predictions_scaled, list):
            predictions_scaled_df = pd.DataFrame(predictions_scaled, columns=['prediction'])
        else: # Assuming it's a single scalar value
            predictions_scaled_df = pd.DataFrame([[predictions_scaled]], columns=['prediction'])

        predictions_actual = self.processor.inverse_transform(predictions_scaled_df)

        return pd.DataFrame(predictions_actual, columns=["prediction"])