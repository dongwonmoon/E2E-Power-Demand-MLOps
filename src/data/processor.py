import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import datetime as dt


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.scaler = MinMaxScaler()

    def _combine_raw_data(self):
        print("Combining raw data files...")
        data_config = self.config["data_processing"]
        raw_data_dir = data_config["raw_data_dir"]
        glob_pattern = data_config["raw_glob_pattern"]
        output_path = os.path.join(raw_data_dir, data_config["combined_raw_filename"])

        file_paths = [
            os.path.join(raw_data_dir, f)
            for f in os.listdir(raw_data_dir)
            if f.endswith(".csv") and f.startswith("power_demand_")
        ]

        if not file_paths:
            print(f"No files found for pattern: {glob_pattern} in {raw_data_dir}")
            return

        df_list = [pd.read_csv(file, encoding="EUC-KR") for file in file_paths]
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        print(f"Combined data saved to {output_path}")
        return combined_df

    def process_raw_data(self):
        """01_process_data.py에서 호출됩니다."""
        print("--- Starting Data Processing ---")
        data_config = self.config["data_processing"]

        # 1. Raw CSVs 결합
        df = self._combine_raw_data()
        if df is None:
            print("No data to process. Exiting.")
            return

        # 2. 데이터 정제 및 분할
        df["날짜"] = pd.to_datetime(df["날짜"])

        # 2020-01-01에 시간을 더하는 작업
        dates = pd.concat([df["날짜"]] * 24).sort_values().reset_index(drop=True)
        hour_cols = [f"{i}시" for i in range(1, 25)]
        demands = df[hour_cols].stack().reset_index().iloc[:, 1:]

        df = pd.concat([dates, demands], axis=1)
        df.columns = ["날짜", "시간", "demand"]

        def apply_hour(x):
            date = x.날짜
            return dt.datetime(
                year=date.year,
                month=date.month,
                day=date.day,
                hour=int(x["시간"][:-1]) - 1,
            )

        df["날짜"] = df[["날짜", "시간"]].apply(apply_hour, axis=1)
        df = df.drop(columns="시간").rename(columns={"날짜": "date"})

        split_date = dt.datetime.strptime(data_config["split_date"], "%Y-%m-%d")
        train_df = df[df["date"] < split_date]
        stream_df = df[df["date"] >= split_date]

        # 3. 처리된 데이터 저장
        processed_dir = data_config["processed_data_dir"]
        os.makedirs(processed_dir, exist_ok=True)
        train_df.to_csv(
            os.path.join(processed_dir, data_config["train_filename"]), index=False
        )
        stream_df.to_csv(
            os.path.join(processed_dir, data_config["stream_filename"]), index=False
        )
        print("--- Data Processing Finished ---")

    def fit_scaler(self, data: pd.DataFrame) -> None:
        """학습 데이터로 스케일러를 학습시킵니다."""
        self.scaler.fit(data[["demand"]])

    def save_scaler(self, path: str) -> None:
        """학습된 스케일러를 저장합니다."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str) -> None:
        """저장된 스케일러를 불러옵니다."""
        self.scaler = joblib.load(path)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        데이터를 스케일링하고 차분 피처를 생성합니다.
        """
        if "demand" not in data.columns:
            raise ValueError("Input DataFrame must contain a 'demand' column.")

        # 스케일링
        data["demand_scaled"] = self.scaler.transform(data[["demand"]])

        # 차분
        data["demand_diff"] = data["demand_scaled"].diff()

        # 차분으로 인해 첫 행에 NaN이 생기므로 제거
        return data.dropna().reset_index(drop=True)

    def inverse_transform(self, scaled_prediction: pd.DataFrame) -> pd.DataFrame:
        """스케일링된 예측값을 원래 스케일로 되돌립니다."""
        # 역변환을 위해 올바른 shape의 더미 배열 생성
        # 스케일러는 [n_samples, n_features] 형태를 기대하므로
        # 예측값을 첫 번째 열로 하는 2D 배열을 만듭니다.
        return self.scaler.inverse_transform(scaled_prediction)
