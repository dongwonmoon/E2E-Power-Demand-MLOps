from kafka import KafkaConsumer
import json


class Consumer:
    def __init__(self, topic):
        self.topic = topic
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=["localhost:9092"],
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )


if __name__ == "__main__":
    consumer = Consumer("power_demand")
