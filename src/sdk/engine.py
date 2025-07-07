# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
from constraint_lattice.compiler import ConstraintOntologyCompiler
from constraint_lattice.engine.apply import apply_constraints
from constraint_lattice.engine.loader import load_constraints_from_yaml
from constraint_lattice.engine import Constraint
from typing import Optional, List

class ConstraintEngine:
    """
    Engine for applying constraints to text.

    Args:
        config_path: Path to the configuration YAML file.
        profile: Profile name to load from the configuration.
        search_modules: List of modules to search for constraints.
        constraints: Optional list of constraint instances to use directly.
    """

    def __init__(
        self,
        config_path: Optional[str] = "constraints.yaml",
        profile: Optional[str] = "default",
        search_modules: Optional[List[str]] = None,
        constraints: Optional[List[Constraint]] = None,
        use_compiler: bool = False,
    ):
        self.config_path = config_path
        self.profile = profile
        self.search_modules = search_modules or []
        self.compiled_constraints = []
        if use_compiler:
            compiler = ConstraintOntologyCompiler()
            try:
                self.compiled_constraints = compiler.compile_file(config_path)
            except FileNotFoundError:
                self.compiled_constraints = []

        if constraints is None:
            self.constraints = load_constraints_from_yaml(
                config_path, profile
            )
        else:
            self.constraints = constraints

    def run(self, prompt: str, output: str, return_trace: bool = False):
        return apply_constraints(
            prompt, output, self.constraints, return_trace=return_trace
        )

    def get_compiled_constraints(self):
        """Return compiled constraints if available."""
        return self.compiled_constraints

