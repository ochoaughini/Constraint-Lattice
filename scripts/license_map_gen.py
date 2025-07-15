# SPDX-License-Identifier: MIT
# (c) 2025 ochoaughini. See LICENSE for full terms.
"""Wrapper script to generate LICENSE_MAP.md.

This script simply delegates to the canonical `src/tools/gen_license_map.py`
so it can be invoked from the `scripts/` directory or via CI workflows.
"""

from src.tools.gen_license_map import main

if __name__ == "__main__":
    main()
