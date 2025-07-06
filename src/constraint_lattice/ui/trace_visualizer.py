# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import streamlit as st
from kafka import KafkaConsumer
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from difflib import SequenceMatcher
import hashlib
import time
from kafka.errors import NoBrokersAvailable
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_BROKER = "localhost:9092"  # Accessible from host
TOPIC_NAME = "constraint_audit"

# Flag to enable/disable Kafka
USE_KAFKA = False  # Set to False to use sample data

@st.cache_resource
def get_kafka_consumer():
    if not USE_KAFKA:
        return None
    
    max_retries = 5
    retry_delay = 5  # seconds
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt+1}/{max_retries}: Connecting to Kafka at {KAFKA_BROKER}")
            
            # Test connection first
            consumer = KafkaConsumer(bootstrap_servers=KAFKA_BROKER)
            consumer.topics()  # This forces a connection attempt
            
            logger.info("Successfully connected to Kafka")
            return KafkaConsumer(
                TOPIC_NAME,
                bootstrap_servers=KAFKA_BROKER,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                api_version=(2, 0, 2)  # Specify API version explicitly
            )
        except NoBrokersAvailable as e:
            logger.error(f"Kafka connection failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("All connection attempts failed")
                raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

def render_token_diff(original: str, transformed: str) -> None:
    """Render token-level differences between original and transformed text."""
    matcher = SequenceMatcher(None, original.split(), transformed.split())
    html_output = ""

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            html_output += " ".join(original.split()[a0:a1]) + " "
        elif opcode == "insert":
            html_output += f"<span style='background-color:#d0f0c0;'> {' '.join(transformed.split()[b0:b1])} </span> "
        elif opcode == "delete":
            html_output += f"<span style='background-color:#f0d0d0;text-decoration:line-through;'> {' '.join(original.split()[a0:a1])} </span> "
        elif opcode == "replace":
            html_output += (
                f"<span style='background-color:#f0d0d0;text-decoration:line-through;'> {' '.join(original.split()[a0:a1])} </span> "
                f"<span style='background-color:#d0f0c0;'> {' '.join(transformed.split()[b0:b1])} </span> "
            )

    st.markdown(html_output, unsafe_allow_html=True)

# Core layout
st.set_page_config(layout="wide")
st.title("Constraint Lattice Trace Visualizer")

# Session filter pane
with st.sidebar:
    st.header("Session Filters")
    session_id = st.selectbox("Session ID", options=["session1", "session2"])
    date_range = st.slider(
        "Date Range", 
        min_value=datetime.now()-timedelta(days=7), 
        max_value=datetime.now(),
        value=(datetime.now()-timedelta(days=1), datetime.now())
    )
    evaluator_types = st.multiselect("Evaluator Types", options=["semantic", "regex", "symbolic"])
    constraint_filter = st.text_input("Constraint ID Filter")

# Add session caching
def load_sample_traces() -> List[Dict]:
    """Fallback function to load sample trace data"""
    return [
        {
            "session_id": "sample_session",
            "step_id": "step_1",
            "original_text": "Sample original text",
            "transformed_text": "Sample transformed text",
            "constraints": ["sample_constraint"],
            "model": "sample_model",
            "timestamp": "2023-01-01T00:00:00Z"
        }
    ]

@st.cache_resource
def load_session_traces(session_id: str) -> List[Dict]:
    if not USE_KAFKA:
        st.warning("Kafka is disabled. Using sample data.")
        return load_sample_traces()
    
    try:
        consumer = get_kafka_consumer()
        if consumer is None:
            raise Exception("Kafka consumer not available")
        
        steps = []
        for message in consumer:
            if message.value['session_id'] == session_id:
                steps.append(message.value)
        return steps
    except Exception as e:
        st.error(f"Failed to load traces from Kafka: {str(e)}")
        st.info("Loading sample data instead...")
        return load_sample_traces()

# Add fingerprint validation
def validate_fingerprint(step):
    current_hash = hashlib.sha256(json.dumps(step['config']).encode()).hexdigest()
    return current_hash == step['config_hash']

# Main trace viewer
st.header("Audit Trace")
steps = load_session_traces(session_id)
for index, step in enumerate(steps):
    if validate_fingerprint(step):
        with st.expander(f"Step {index}:"):
            render_token_diff(step['pre_text'], step['post_text'])
    else:
        st.error(f"Invalid fingerprint for step {index}")

# Metadata inspector
with st.expander("Step Metadata"):
    st.json({"key": "value"})

# Diagnostics section
st.header("Trace Diagnostics")
st.plotly_chart(px.bar(x=[1,2,3], y=[4,5,6]))
