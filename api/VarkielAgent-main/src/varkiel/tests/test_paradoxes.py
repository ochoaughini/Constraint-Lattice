import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from central_controller import CentralController
from constraint_lattice_adapter import ConstraintLatticeAdapter
from semantic_resonance import ResonanceFilter as SemanticResonanceEngine
from structural_constraint_engine import StructuralConstraintEngine
from risk_balancer import RiskBalancer
from state_vector import StateVector
import numpy as np

# Mock classes for the components for testing
class MockConstraintLattice(ConstraintLatticeAdapter):
    def apply(self, state: np.ndarray) -> StateVector:
        return StateVector(state, coherence=0.8)

class MockStructuralEngine(StructuralConstraintEngine):
    def apply_constraints(self, state: np.ndarray) -> StateVector:
        return StateVector(state, coherence=0.7)

class MockPhenomenologicalTracker:
    def track(self, state):
        return state

class MockRecursiveInvarianceMonitor:
    def monitor(self, state):
        return state

@pytest.fixture
def controller():
    # Create a controller with mock components
    return CentralController(
        structural_engine=MockStructuralEngine(),
        coherence_engine=SemanticResonanceEngine(),
        phenomenological_tracker=MockPhenomenologicalTracker(),
        recursive_invariance_monitor=MockRecursiveInvarianceMonitor(),
        risk_balancer=RiskBalancer()
    )

@pytest.mark.parametrize("paradox", [
    "This statement is false",
    "What happens when an unstoppable force meets an immovable object?",
    "Can God create a stone so heavy that He cannot lift it?"
])
def test_paradox_handling(controller, paradox):
    response = controller.process_query(paradox)
    # We expect the system to either suspend or recoil
    assert "[SUSPENDED]" in response or "[RECOIL]" in response
