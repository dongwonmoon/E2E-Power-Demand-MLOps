from kafka import KafkaProducer
import json
import pandas as pd
import time

df = pd.read_csv("./data/processed/stream.csv")

producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

if __name__ == "__main__":
    for index, row in df.iterrows():
        msg = row.to_dict()
        producer.send("power_demand", msg)
        time.sleep(1)
        print(f"[Producer] Send: {msg}")
