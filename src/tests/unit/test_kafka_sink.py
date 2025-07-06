# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os
from unittest.mock import patch, MagicMock
from constraint_lattice.engine.audit_sink_kafka import KafkaAuditSink


def test_kafka_sink_publish():
    """Test Kafka sink publish"""
    with patch('confluent_kafka.Producer') as mock_producer:
        # Mock environment
        os.environ["ENABLE_KAFKA_SINK"] = "true"
        os.environ["KAFKA_SERVERS"] = "localhost:9092"
        
        sink = KafkaAuditSink()
        mock_producer.assert_called_with({"bootstrap.servers": "localhost:9092"})
        
        # Test publish
        test_step = {"constraint_id": "test"}
        sink.publish(test_step)
        
        # Check that produce was called
        sink.producer.produce.assert_called_once()
        
        # Test flush
        sink.flush()
        sink.producer.flush.assert_called_once()
