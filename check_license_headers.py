# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.

import os
import sys

# Define the expected headers for each license type and comment style
HEADERS = {
    'MIT': {
        '.py': '# SPDX-License-Identifier: MIT\n# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.',
        '.js': '// SPDX-License-Identifier: MIT\n// Copyright (c) 2025 ochoaughini. See LICENSE for full terms.',
        '.ts': '// SPDX-License-Identifier: MIT\n// Copyright (c) 2025 ochoaughini. See LICENSE for full terms.',
        '.css': '/* SPDX-License-Identifier: MIT */\n/* Copyright (c) 2025 ochoaughini. See LICENSE for full terms. */',
        '.html': '<!-- SPDX-License-Identifier: MIT -->\n<!-- Copyright (c) 2025 ochoaughini. See LICENSE for full terms. -->'
    },
    'BSL': {
        '.py': '# SPDX-License-Identifier: BSL-1.1\n# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.',
        '.js': '// SPDX-License-Identifier: BSL-1.1\n// Copyright (c) 2025 ochoaughini. See LICENSE for full terms.',
        '.ts': '// SPDX-License-Identifier: BSL-1.1\n// Copyright (c) 2025 ochoaughini. See LICENSE for full terms.',
        '.css': '/* SPDX-License-Identifier: BSL-1.1 */\n/* Copyright (c) 2025 ochoaughini. See LICENSE for full terms. */',
        '.html': '<!-- SPDX-License-Identifier: BSL-1.1 -->\n<!-- Copyright (c) 2025 ochoaughini. See LICENSE for full terms. -->'
    }
}

def check_license_headers():
    repo_root = os.getcwd()
    errors = []
    for root, dirs, files in os.walk(repo_root):
        # Skip hidden directories
        if any(dir.startswith('.') for dir in root.split(os.sep)):
            continue
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext not in HEADERS['MIT']:
                continue
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, repo_root)
            # Determine expected license based on directory
            if relative_path.startswith('saas') or relative_path.startswith('api') or relative_path.startswith('billing'):
                expected_spdx = HEADERS['BSL'][ext]
            else:
                expected_spdx = HEADERS['MIT'][ext]
            with open(file_path, 'r') as f:
                content = f.read(500)
                if expected_spdx not in content:
                    errors.append(f"Missing or incorrect license header in {relative_path}")
    if errors:
        print("\n".join(errors))
        sys.exit(1)

if __name__ == '__main__':
    check_license_headers()
