# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os
import requests
from typing import Literal

ExecutionMode = Literal["supervisory", "executor"]

def get_execution_mode() -> ExecutionMode:
    """Determine operational mode based on LLM endpoint availability"""
    # Check environment override first
    if os.getenv("CONSTRAINT_LATTICE_FORCE_EXECUTOR_MODE", "false").lower() == "true":
        return "executor"
        
    # Check if we have an endpoint configured
    endpoint = os.getenv("LLM_ENDPOINT")
    if not endpoint:
        return "executor"
    
    # Ping the endpoint to check availability
    try:
        response = requests.head(f"{endpoint}/health", timeout=1.0)
        if response.status_code == 200:
            return "supervisory"
    except (requests.exceptions.ConnectionError, 
            requests.exceptions.Timeout):
        pass
        
    return "executor"
