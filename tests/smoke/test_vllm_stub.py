# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""Ensure the codebase behaves gracefully when vllm is absent.

This runs in CPU-only CI where the *vllm* wheel is unlikely to be present.
We assert that requesting the backend raises a clear RuntimeError instead of
import-time crashes.
"""

import importlib.util

import pytest

pytest.importorskip("constraints.phi2_backend")
from constraints.phi2_backend import VLLMBackend  # noqa: E402


def test_vllm_backend_guard():
    # Skip if vllm is actually available (full backend tested elsewhere).
    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("Real vllm wheel present; stub guard not applicable")
    with pytest.raises(RuntimeError):
        VLLMBackend(model_name="microsoft/phi-2")
