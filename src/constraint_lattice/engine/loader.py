# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
import hashlib
import importlib
import json
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Type, Union

from .schema import ConstraintConfig, ConstraintSchema
from constraint_lattice.engine import Constraint


def load_constraint_class(class_name: str) -> Type[Constraint]:
    """
    Load a constraint class by name.
    
    Args:
        class_name: Fully qualified name of the constraint class.
        
    Returns:
        The constraint class.
        
    Raises:
        ImportError: If the class cannot be found.
    """
    try:
        module_name, class_name = class_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Could not load constraint class {class_name}: {e}")


def load_constraints_from_yaml(
    yaml_path: str, profile: str = "default"
) -> List[Constraint]:
    """
    Load constraints from a YAML configuration file.
    
    Args:
        yaml_path: Path to the YAML file.
        profile: Name of the profile to load.
        
    Returns:
        List of constraint instances.
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    
    if profile not in config["profiles"]:
        raise ValueError(f"Profile '{profile}' not found in configuration")
        
    raw_entries = config["profiles"][profile]
    constraints = []
    
    for entry in raw_entries:
        if isinstance(entry, str):
            class_name = entry
            params = {}
        elif isinstance(entry, dict):
            class_name = entry["class"]
            params = entry.get("params", {})
        else:
            raise ValueError(f"Invalid entry type in profile: {type(entry)}")
            
        constraint_cls = load_constraint_class(class_name)
        constraint = constraint_cls(**params)
        constraints.append(constraint)
        
    return constraints


def load_constraints_from_file(path: str) -> List[ConstraintSchema]:
    """
    Load constraints from a YAML or JSON file
    
    Args:
        path: Path to constraint configuration file
        
    Returns:
        List of validated ConstraintSchema objects
    """
    # Read file
    with open(path, 'r') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            config_data = yaml.safe_load(f)
        elif path.endswith('.json'):
            config_data = json.load(f)
        else:
            raise ValueError("Unsupported file format")
            
    # Parse into ConstraintConfig
    config = ConstraintConfig(**config_data)
    
    # Process includes
    if hasattr(config, 'includes') and config.includes:
        included_constraints = []
        for include_path in config.includes:
            # Resolve relative paths
            if not os.path.isabs(include_path):
                include_path = os.path.join(os.path.dirname(path), include_path)
            included_constraints.extend(load_constraints_from_file(include_path))
        
        # Merge constraints (deduplicate by name)
        constraint_map = {c.name: c for c in config.constraints}
        for constraint in included_constraints:
            if constraint.name not in constraint_map:
                constraint_map[constraint.name] = constraint
        config.constraints = list(constraint_map.values())
    
    # Compute input hash for each constraint
    for constraint in config.constraints:
        constraint_json = json.dumps(constraint.dict(), sort_keys=True)
        constraint.input_hash = hashlib.sha256(constraint_json.encode()).hexdigest()
    
    return config.constraints
