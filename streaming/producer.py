from kafka import KafkaProducer
import json
import pandas as pd
import time
from datetime import timedelta

# Load data and convert date column to datetime objects
df = pd.read_csv("./data/processed/stream.csv")
df["date"] = pd.to_datetime(df["date"])

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    # datetime 객체를 json으로 보내기 위해 default=str 옵션 추가
    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
)

if __name__ == "__main__":
    # 각 데이터에 순차적으로 시간을 더해 고유한 타임스탬프 생성
    for index, row in df.iterrows():
        # 원본 날짜에 시간(index)을 더함
        new_timestamp = row["date"] + timedelta(hours=index)

        msg = row.to_dict()
        # 기존 date 필드를 새로 만든 타임스탬프로 교체
        msg["date"] = new_timestamp.isoformat()

        producer.send("power_demand", msg)
        time.sleep(10)
        print(f"[Producer] Send: {msg}")
