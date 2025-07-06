# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import streamlit as st
import time
from .audit_data_loader import AuditDataLoader

# Initialize data loader
data_loader = AuditDataLoader()

# Page title
st.title("Constraint Lattice Audit Panel")

# Sidebar filters
st.sidebar.header("Filters")
constraint_id = st.sidebar.text_input("Constraint ID")
evaluator_type = st.sidebar.selectbox("Evaluator Type", ["", "regex", "text", "semantic", "symbolic"])
min_confidence = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.0)

# Session selector
sessions = data_loader.get_sessions()
selected_session = st.selectbox("Select Session", sessions)

# Load session steps
if selected_session:
    steps = data_loader.get_session_steps(selected_session)
    
    # Display steps
    for step in steps:
        # Apply filters
        if constraint_id and constraint_id != step.get("constraint_id"):
            continue
        if evaluator_type and evaluator_type != step.get("method"):
            continue
        if step.get("confidence_score", 0) < min_confidence:
            continue
            
        # Display step
        with st.expander(f"{step['constraint_name']} - {step['method']}"):
            st.json(step)
            
            # Show text diff
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input")
                st.code(step["pre_text"])
            with col2:
                st.subheader("Output")
                st.code(step["post_text"])
            
            # Show metadata
            st.caption(f"Elapsed: {step['elapsed_ms']}ms | Confidence: {step.get('confidence_score', 'N/A')}")

# Real-time toggle
real_time = st.checkbox("Enable real-time updates")

if real_time:
    # Auto-refresh every 2 seconds
    while True:
        new_steps = data_loader.get_new_steps()
        if new_steps:
            st.experimental_rerun()
        time.sleep(2)
