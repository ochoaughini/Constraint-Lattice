import os
import json
import uuid
from confluent_kafka import Producer
from typing import Dict


class KafkaAuditSink:
    """Publishes audit steps to Kafka"""
    
    def __init__(self):
        self.bootstrap_servers = os.getenv("KAFKA_SERVERS", "localhost:9092")
        self.topic = os.getenv("KAFKA_TOPIC", "constraint_audit")
        self.producer = Producer({"bootstrap.servers": self.bootstrap_servers})
        
    def publish(self, audit_step: Dict):
        """Publish audit step to Kafka"""
        # Enrich step with session id and timestamp
        payload = {
            "session_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.utcnow().isoformat(),
            **audit_step
        }
        
        # Serialize to JSON
        value = json.dumps(payload)
        
        # Publish asynchronously
        self.producer.produce(self.topic, value=value)
        self.producer.poll(0)
        
    def flush(self):
        """Flush pending messages"""
        self.producer.flush()


def get_kafka_sink():
    """Get Kafka sink if enabled"""
    if os.getenv("ENABLE_KAFKA_SINK", "false").lower() == "true":
        return KafkaAuditSink()
    return None
