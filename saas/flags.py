# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.

Importing this module from MIT-licensed packages is safe; it exposes only a
boolean constant and never pulls additional commercial dependencies.
"""
from __future__ import annotations

import os

ENABLED: bool = os.getenv("ENABLE_SAAS_FEATURES", "").lower() == "true"

__all__ = ["ENABLED"]
