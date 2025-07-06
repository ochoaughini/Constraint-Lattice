import os
import re

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

for root, dirs, files in os.walk('.'):
    # Rename directories first
    for name in dirs:
        new_name = to_snake_case(name)
        if new_name != name:
            os.rename(os.path.join(root, name), os.path.join(root, new_name))
    
    # Rename files
    for name in files:
        new_name = to_snake_case(name)
        if new_name != name:
            os.rename(os.path.join(root, name), os.path.join(root, new_name))
