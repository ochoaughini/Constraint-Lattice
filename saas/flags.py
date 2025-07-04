"""Feature flag helper for SaaS / BSL code.

Importing this module from MIT-licensed packages is safe; it exposes only a
boolean constant and never pulls additional commercial dependencies.
"""
from __future__ import annotations

import os

ENABLED: bool = os.getenv("ENABLE_SAAS_FEATURES", "").lower() == "true"

__all__ = ["ENABLED"]
