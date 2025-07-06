# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.

This module is only available when the environment variable
``ENABLE_SAAS_FEATURES=true`` is present. Attempting to import it without the
flag raises ``ImportError`` so that core (MIT) builds remain self-contained and
immune from accidental BSL leakage.
"""

import os

if not os.getenv("ENABLE_SAAS_FEATURES", "false").lower() in ["true", "1"]:
    raise ImportError(
        "SaaS features are disabled. Set ENABLE_SAAS_FEATURES=true to enable this module."
    )


from functools import lru_cache

from sdk.engine import ConstraintEngine

# Lazily initialise a reusable engine to avoid model reload on every request.


@lru_cache(maxsize=1)
def get_engine() -> ConstraintEngine:
    return ConstraintEngine()
