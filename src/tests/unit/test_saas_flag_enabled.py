# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Ensure SaaS package can be imported when the flag is enabled."""
from __future__ import annotations

import importlib
import os
import sys
import pathlib

# Ensure project root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def test_import_saas_with_flag(monkeypatch):
    monkeypatch.setenv("ENABLE_SAAS_FEATURES", "true")
    saas = importlib.import_module("saas")
    assert hasattr(saas, "get_engine")
