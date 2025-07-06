# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import json
import requests
from difflib import ndiff
import streamlit as st
import websockets
import asyncio
import threading
from pydantic import BaseModel, ValidationError
import os
from pathlib import Path

WORKSPACE_DIR = Path("workspaces")
WORKSPACE_DIR.mkdir(exist_ok=True)

class ConstraintAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def apply_constraints(self, text, constraints=None):
        try:
            response = requests.post(
                f"{self.base_url}/api/constraints/apply",
                json={"text": text, "constraints": constraints or []}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None

def load_audit_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def diff_text(a, b):
    return "\n".join(ndiff(a.split(), b.split()))

class RealtimeMonitor:
    def __init__(self, ws_url="ws://localhost:8000/ws/trace"):
        self.ws_url = ws_url
        self.latest_trace = None
        
    async def listen(self):
        async with websockets.connect(self.ws_url) as ws:
            while True:
                message = await ws.recv()
                self.latest_trace = json.loads(message)
                
    def start(self):
        def run():
            asyncio.run(self.listen())
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

class ConstraintSchema(BaseModel):
    name: str
    type: str
    params: dict = {}
    
def validate_constraints(constraint_json: str) -> bool:
    try:
        constraints = json.loads(constraint_json)
        if not isinstance(constraints, list):
            constraints = [constraints]
            
        for c in constraints:
            ConstraintSchema(**c)
        return True
    except (json.JSONDecodeError, ValidationError) as e:
        st.error(f"Constraint validation error: {str(e)}")
        return False

def save_workspace(name: str, text: str, constraints: str):
    workspace = WORKSPACE_DIR / f"{name}.json"
    with open(workspace, "w") as f:
        json.dump({"text": text, "constraints": constraints}, f)

def load_workspace(name: str) -> dict:
    workspace = WORKSPACE_DIR / f"{name}.json"
    if workspace.exists():
        with open(workspace) as f:
            return json.load(f)
    return {}

# Main Application
api = ConstraintAPI()
monitor = RealtimeMonitor()
monitor.start()

st.title("Constraint Lattice Explorer")

with st.sidebar:
    st.header("Workspace")
    workspace_name = st.text_input("Workspace Name")
    
    if st.button("Save Workspace") and workspace_name:
        save_workspace(workspace_name, text, constraints)
        st.success(f"Saved workspace '{workspace_name}'")
        
    workspace_files = [f.stem for f in WORKSPACE_DIR.glob("*.json")]
    selected_workspace = st.selectbox("Load Workspace", [""] + workspace_files)
    
    if selected_workspace:
        workspace = load_workspace(selected_workspace)
        text = workspace.get("text", "")
        constraints = json.dumps(workspace.get("constraints", []), indent=2)

tab1, tab2 = st.tabs(["Live Constraints", "Audit Logs"])

with tab1:
    text = st.text_area("Input Text", height=200, value=text)
    constraints = st.text_area("Constraints (JSON)", height=100, value=constraints)
    
    if st.button("Apply Constraints") and validate_constraints(constraints):
        try:
            constraints_json = json.loads(constraints)
            result = api.apply_constraints(text, constraints_json)
            
            if result:
                st.subheader("Result")
                st.text_area("Output", result.get("result", ""), height=200)
                
                st.subheader("Execution Trace")
                for step in result.get("steps", []):
                    with st.expander(f"{step['constraint']} ({step['elapsed_ms']}ms)"):
                        st.json(step)
        except json.JSONDecodeError:
            st.error("Invalid JSON format for constraints")
            
    if monitor.latest_trace:
        st.write("Latest trace update:")
        st.json(monitor.latest_trace)

with tab2:
    log_file = st.file_uploader("Upload Audit Log", type="jsonl")
    if log_file:
        log_path = "/tmp/audit_log.jsonl"
        with open(log_path, "wb") as f:
            f.write(log_file.read())
        audit = load_audit_log(log_path)
        
        st.write(f"Loaded {len(audit)} steps")
        for entry in audit:
            with st.expander(f"{entry['constraint']} - {entry['method']}"):
                st.json(entry)
