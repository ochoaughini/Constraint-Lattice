#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Kill any process on port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null

# Start server with auto-reload
uvicorn api.main:app --reload --port 8000
