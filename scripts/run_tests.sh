#!/bin/bash
source venv/bin/activate
export PYTHONPATH=src:$PYTHONPATH
pytest
