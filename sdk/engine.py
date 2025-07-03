from constraint_lattice.engine.apply import apply_constraints
from constraint_lattice.engine.loader import load_constraints_from_yaml


class ConstraintEngine:
    def __init__(
        self, config_path="constraints.yaml", profile="default", search_modules=None
    ):
        if search_modules is None:
            search_modules = [
                "constraint_lattice.constraints.boundary_prime",
                "constraint_lattice.constraints.mirror_law",
                "constraint_lattice.constraints.reset_pulse",
                "constraint_lattice.constraints.constraint_profanity_filter",
                "constraint_lattice.constraints.phi2_moderation",
                "constraint_lattice.constraints.semantic_similarity_guard",
            ]
        self.constraints = []  # Temporarily bypass YAML loading to test application startup

    def run(self, prompt: str, output: str, return_trace: bool = False):
        return apply_constraints(
            prompt, output, self.constraints, return_trace=return_trace
        )
