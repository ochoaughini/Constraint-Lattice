# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os
import sys
import re
import re
import clean_old_headers  # Import the module

MIT_HEADER = """SPDX-License-Identifier: MIT
Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""

BSL_HEADER = """SPDX-License-Identifier: BSL-1.1
Copyright (c) 2025 Lexsight LCC. All rights reserved.
See saas/LICENSE-BSL.txt for full terms.
"""

# Map of file extensions to their comment style
COMMENT_STYLES = {
    '.py': '#',
    '.js': '//',
    '.ts': '//',
    '.css': '/*',
    '.html': '<!--'
}

def get_header(comment_style, header_text):
    """Format header text with appropriate comment style.
    
    Args:
        comment_style: The comment prefix for the file type
        header_text: Raw header text to format
    
    Returns:
        Formatted header string
    """
    # Handle block comment styles
    if comment_style == '/*':
        return f"/*\n{header_text}*/\n"
    elif comment_style == '<!--':
        return f"<!--\n{header_text}-->\n"
    
    # Handle line comment styles
    lines = header_text.split('\n')
    formatted_lines = []
    
    for line in lines:
        if line.strip():  # Only add non-empty lines
            formatted_lines.append(f"{comment_style} {line}")
    
    return '\n'.join(formatted_lines) + '\n'

def is_text_file(file_path):
    with open(file_path, 'rb') as f:
        chunk = f.read(1024)
        if b'\0' in chunk:
            return False
    return True

def add_header_to_file(file_path, header):
    if not os.path.isfile(file_path):
        print(f"Skipping non-file: {file_path}")
        return

    # Skip binary files
    if not is_text_file(file_path):
        print(f"Skipping binary file: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove any existing SPDX header lines to avoid duplicates
    # This targets lines that start with a comment character and then SPDX-License-Identifier
    content = re.sub(r'^\s*[#/]+\s*SPDX-License-Identifier:.*$\n?', '', content, flags=re.MULTILINE)

    # Check if the exact header is already at the beginning
    if content.startswith(header):
        print(f"Header already present in {file_path}")
        return

    # Add the header
    new_content = header + content

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Added header to {file_path}")

def main():
    # Clean old headers
    clean_old_headers.main()
    
    repo_root = os.getcwd()
    for root, dirs, files in os.walk(repo_root):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in COMMENT_STYLES:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_root)
                comment_style = COMMENT_STYLES[ext]
                
                print(f"Processing: {relative_path}")
                
                if relative_path.startswith('saas') or relative_path.startswith('api') or relative_path.startswith('billing'):
                    header_text = get_header(comment_style, BSL_HEADER)
                else:
                    header_text = get_header(comment_style, MIT_HEADER)
                
                add_header_to_file(file_path, header_text)

if __name__ == '__main__':
    main()
