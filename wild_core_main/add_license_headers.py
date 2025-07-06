# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import os
import sys

MIT_HEADER = """# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
"""

BSL_HEADER = """# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
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
    if comment_style == '/*':
        return f"/*\n{header_text}*/\n"
    elif comment_style == '<!--':
        return f"<!--\n{header_text}-->\n"
    else:
        lines = header_text.split('\n')
        return '\n'.join(f"{comment_style} {line}" for line in lines if line) + '\n'

def add_header_to_file(file_path, header, comment_style):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Skip if already has a header
    if content.startswith(header):
        return
    
    # For CSS and HTML, we have multi-line comments
    if comment_style in ['/*', '<!--']:
        # We'll look for the first occurrence of the comment style to avoid false positives
        if content.startswith(comment_style):
            return
    else:
        # For single-line comments, check the first line
        first_line = content.split('\n')[0]
        if first_line.startswith(comment_style):
            return
    
    with open(file_path, 'w') as f:
        f.write(header + content)

def main():
    repo_root = os.getcwd()
    for root, dirs, files in os.walk(repo_root):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in COMMENT_STYLES:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_root)
                comment_style = COMMENT_STYLES[ext]
                
                if relative_path.startswith('saas') or relative_path.startswith('api') or relative_path.startswith('billing'):
                    header_text = get_header(comment_style, BSL_HEADER)
                else:
                    header_text = get_header(comment_style, MIT_HEADER)
                
                add_header_to_file(file_path, header_text, comment_style)

if __name__ == '__main__':
    main()
