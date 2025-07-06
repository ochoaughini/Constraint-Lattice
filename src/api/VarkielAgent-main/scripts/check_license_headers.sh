#!/bin/bash

# SPDX-License-Identifier: AGPL-3.0-only OR Commercial

# Check for license headers in all Python files
find src -name "*.py" | while read file; do
    if ! grep -q "SPDX-License-Identifier: AGPL-3.0-only OR Commercial" "$file"; then
        echo "Missing license header in $file"
        exit 1
    fi
done

echo "All files have license headers"
exit 0
