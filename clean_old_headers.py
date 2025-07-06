# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os
import re

def clean_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove malformed headers: lines starting with any number of comment characters followed by SPDX-License-Identifier
    # This targets headers that were added with extra '#' or '//' characters
    cleaned_content = re.sub(r'^[\s]*[#/]+\s*SPDX-License-Identifier:.*$\n?', '', content, flags=re.MULTILINE)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

def main():
    # Walk through all directories and files
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith(('.py', '.js', '.html')):
                file_path = os.path.join(root, file)
                clean_file(file_path)

if __name__ == "__main__":
    main()
