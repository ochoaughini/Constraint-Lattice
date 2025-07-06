import os
import sys

# Define the expected headers for each license type and comment style
HEADERS = {
    'MIT': {
        '.py': '# SPDX-License-Identifier: MIT',
        '.js': '// SPDX-License-Identifier: MIT',
        '.ts': '// SPDX-License-Identifier: MIT',
        '.css': '/* SPDX-License-Identifier: MIT */',
        '.html': '<!-- SPDX-License-Identifier: MIT -->'
    },
    'BSL': {
        '.py': '# SPDX-License-Identifier: BSL-1.1',
        '.js': '// SPDX-License-Identifier: BSL-1.1',
        '.ts': '// SPDX-License-Identifier: BSL-1.1',
        '.css': '/* SPDX-License-Identifier: BSL-1.1 */',
        '.html': '<!-- SPDX-License-Identifier: BSL-1.1 -->'
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
                expected_header = HEADERS['BSL'][ext]
            else:
                expected_header = HEADERS['MIT'][ext]
            with open(file_path, 'r') as f:
                content = f.read()
                if not content.startswith(expected_header):
                    errors.append(f"Missing or incorrect license header in {relative_path}")
    if errors:
        print("\n".join(errors))
        sys.exit(1)

if __name__ == '__main__':
    check_license_headers()
