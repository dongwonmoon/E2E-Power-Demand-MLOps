import pandas as pd
import datetime as dt


class DataProcessor:
    def __init__(self, config):
        self.config = config

    def process(self):
        print("Processing data...")
        df = df.rename(columns={"날짜": "date", "시간": "demand"})
        df = pd.read_csv(self.config["data"]["combined_raw_path"])

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df[["date", "demand"]]

        # Split data based on date
        train_df = df[df["date"] < dt.datetime(2023, 1, 1)]
        stream_df = df[df["date"] >= dt.datetime(2023, 1, 1)]

        # Save processed data
        train_df.to_csv(self.config["data"]["processed"]["train"], index=False)
        stream_df.to_csv(self.config["data"]["processed"]["stream"], index=False)
        print("Data processing complete. Train and stream data saved.")
