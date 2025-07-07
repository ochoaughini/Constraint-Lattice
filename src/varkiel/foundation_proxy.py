# SPDX-License-Identifier: MIT
"""Minimal interface to external language models."""

from __future__ import annotations

from typing import Iterable


class FoundationProxy:
    """Return canned responses to mimic external LLMs."""

    def __init__(self, responses: Iterable[str] | None = None) -> None:
        self._responses = list(responses or [])

    def query(self, prompt: str) -> str:
        return self._responses.pop(0) if self._responses else ""
