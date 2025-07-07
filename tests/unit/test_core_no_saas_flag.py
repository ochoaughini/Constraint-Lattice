# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.

"""We run these tests without setting the ENABLE_SAAS_FEATURES env var.
Any import of the ``saas`` package should therefore fail, and normal
engine functionality should remain unaffected.
"""
from __future__ import annotations

import os
import sys
import importlib
import pathlib
import pytest

# Ensure project root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# Guarantee flag is off for this test session
os.environ.pop("ENABLE_SAAS_FEATURES", None)


def test_engine_import_without_saas():
    """Importing engine should succeed when SaaS flag disabled."""
    import engine.apply  # noqa: F401 - just ensure import path works


def test_importing_saas_fails():
    """Importing `saas` should raise ImportError when flag is off."""
    with pytest.raises(ImportError):
        importlib.import_module("saas")
