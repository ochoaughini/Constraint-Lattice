# SPDX-License-Identifier: MIT
"""Placeholder for the Gemma formalization model."""

from __future__ import annotations


def extract_facts(text: str) -> list[str]:
    """Very naive fact extractor splitting sentences."""
    return [sent.strip() for sent in text.split('.') if sent.strip()]
