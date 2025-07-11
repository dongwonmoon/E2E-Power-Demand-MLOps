import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.dataset import TimeSeriesDataset


class LSTMModel(nn.Module):
    def __init__(self, params, input_size, output_size):
        super().__init__()
        self.params = params
        self._trained = False

        hidden_size = params.get("hidden_size", 50)
        num_layers = params.get("num_layers", 2)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, data):
        print("Training LSTM Model...")
        seq_len = self.params.get("seq_len", 16)
        dataset = TimeSeriesDataset(data, seq_len)
        batch_size = self.params.get("batch_size", 256)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        learning_rate = self.params.get("learning_rate", 0.005)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        epochs = self.params.get("epochs", 5)

        self.train()
        epoch_losses = []
        for epoch in tqdm(range(epochs), desc="Epoch"):
            total_loss = 0
            for X, y in tqdm(loader, leave=False, desc="Batch"):
                optimizer.zero_grad()
                predictions = self(X)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            epoch_losses.append(avg_loss)

        self._trained = True
        print("LSTM model training complete.")
        return self, epoch_losses

    def predict(self, data):
        if not self._trained:
            raise ValueError("Model has not been trained yet")

        self.eval()
        # Ensure data is a tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        # Add batch dimension if it's missing
        if data.dim() == 2:
            data = data.unsqueeze(0)

        with torch.no_grad():
            predictions = self(data)

        return predictions.squeeze().tolist()  # Return as list
