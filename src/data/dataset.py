from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len):
        self.series = series
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        return x, y
