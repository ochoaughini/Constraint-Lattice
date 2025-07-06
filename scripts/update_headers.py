#!/usr/bin/env python3
"""
Automated license header updater for Constraint-Lattice project
"""

import os
import re
from pathlib import Path

# License header templates
HEADERS = {
    "MIT": (
        "# SPDX-License-Identifier: MIT\n"
        "# Copyright (c) 2025 ochoaughini. All rights reserved.\n"
        "# See LICENSE for full terms.\n\n"
    ),
    "BSL": (
        "# SPDX-License-Identifier: BSL-1.1\n"
        "# Copyright (c) 2025 Lexsight LCC. All rights reserved.\n"
        "# See saas/LICENSE-BSL.txt for full terms.\n\n"
    ),
    "PROPRIETARY": (
        "# SPDX-License-Identifier: PROPRIETARY\n"
        "# Copyright (c) 2025 LXLite LLC. All rights reserved.\n"
        "# See LICENSE_PROPRIETARY for full terms.\n\n"
    )
}

# Directory to license mapping
LICENSE_MAP = {
    "constraint_lattice": "MIT",
    "src/constraint_lattice": "MIT",
    "saas": "BSL",
    "src/clattice": "BSL"
}

# Special proprietary files
PROPRIETARY_FILES = {
    "policy_dsl.py",
    "moderation_filter_phi2.py",
    "semantic_alignment_harness.py"
}

def get_license_type(file_path):
    """Determine license type based on file location and name"""
    # Check if file is proprietary
    if file_path.name in PROPRIETARY_FILES:
        return "PROPRIETARY"
    
    # Check directory mapping
    for path, license_type in LICENSE_MAP.items():
        if path in str(file_path):
            return license_type
    
    return None

def update_file_headers(root_dir):
    """Update headers for all source files in the project"""
    for root, _, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(('.py', '.js', '.ts', '.css')):
                continue
                
            file_path = Path(root) / file
            license_type = get_license_type(file_path)
            
            if not license_type:
                print(f"Skipping {file_path}: No license mapping")
                continue
            
            print(f"Updating {file_path} with {license_type} header")
            
            with open(file_path, 'r+', encoding='utf-8') as f:
                content = f.read()
                
                # Remove existing headers
                content = re.sub(r'^# SPDX-License-Identifier:.*?\n\n', '', content, flags=re.DOTALL)
                
                # Preserve shebang if present
                shebang = ''
                if content.startswith('#!/'):
                    shebang_line = content.split('\n', 1)[0]
                    shebang = shebang_line + '\n'
                    content = content[len(shebang_line):].lstrip('\n')
                
                # Add new header
                new_content = shebang + HEADERS[license_type] + content
                
                # Write back to file
                f.seek(0)
                f.write(new_content)
                f.truncate()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    update_file_headers(os.path.join(project_root, ".."))
    print("Header update complete!")
