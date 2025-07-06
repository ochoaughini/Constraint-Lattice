# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os

# Define the license header for MIT
MIT_HEADER = """# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""

# Define directories to skip
SKIP_DIRS = ["venv", ".venv", "__pycache__", ".git", "node_modules"]

# Define file extensions to process
EXTENSIONS = ['.py', '.js', '.ts', '.css']

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

for root, dirs, files in os.walk(ROOT_DIR):
    # Skip directories
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    
    for file in files:
        if any(file.endswith(ext) for ext in EXTENSIONS):
            file_path = os.path.join(root, file)
            
            # Read the existing content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Prepend the header
            new_content = MIT_HEADER + content
            
            # Write back
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            print(f"Added header to {file_path}")
