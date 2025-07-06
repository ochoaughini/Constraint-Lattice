import os
import pytest
from constraint_lattice.engine.loader import load_constraints_from_file


def test_load_constraints_from_file(tmp_path):
    # Create test config
    config_path = os.path.join(tmp_path, "test.yaml")
    with open(config_path, 'w') as f:
        f.write("""
version: "1.0"
constraints:
  - name: "no_profanity"
    type: "regex"
    params:
      pattern: "[!@#$%^&*()]"
      replacement: ""
""")
    
    constraints = load_constraints_from_file(config_path)
    assert len(constraints) == 1
    assert constraints[0].name == "no_profanity"
    assert constraints[0].type == "regex"
    
    # Test includes
    include_path = os.path.join(tmp_path, "include.yaml")
    with open(include_path, 'w') as f:
        f.write("""
version: "1.0"
constraints:
  - name: "no_emails"
    type: "regex"
    params:
      pattern: "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
      replacement: "[REDACTED]"
""")
    
    config_with_include = os.path.join(tmp_path, "main.yaml")
    with open(config_with_include, 'w') as f:
        f.write(f"""
version: "1.0"
includes:
  - "{include_path}"
constraints:
  - name: "no_profanity"
    type: "regex"
    params:
      pattern: "[!@#$%^&*()]"
      replacement: ""
""")
    
    constraints = load_constraints_from_file(config_with_include)
    assert len(constraints) == 2
    names = {c.name for c in constraints}
    assert "no_profanity" in names
    assert "no_emails" in names
    
    # Test deduplication
    with open(config_with_include, 'w') as f:
        f.write(f"""
version: "1.0"
includes:
  - "{include_path}"
constraints:
  - name: "no_emails"  # Override included constraint
    type: "regex"
    params:
      pattern: "overridden"
      replacement: ""
""")
    
    constraints = load_constraints_from_file(config_with_include)
    assert len(constraints) == 1
    assert constraints[0].name == "no_emails"
    assert constraints[0].params["pattern"] == "overridden"
