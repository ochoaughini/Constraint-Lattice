# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.

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
