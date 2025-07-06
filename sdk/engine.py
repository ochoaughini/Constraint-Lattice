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
    ):
        self.config_path = config_path
        self.profile = profile
        self.search_modules = search_modules or []
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
