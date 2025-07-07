# SPDX-License-Identifier: MIT
"""Placeholder for the Phi-2 moderation model."""

from __future__ import annotations


def moderate(text: str) -> str:
    """Return text with simple profanity masking."""
    return text.replace("badword", "****")
