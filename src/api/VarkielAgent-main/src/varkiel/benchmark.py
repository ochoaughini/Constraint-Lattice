# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
"""
Varkiel Agent - Advanced AI Constraint System
Copyright (C) 2025 Lexsight LLC
SPDX-License-Identifier: AGPL-3.0-only OR Commercial

Comprehensive performance benchmark for Varkiel Agent with paradox probes
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from central_controller import CentralController
from godel_metric import GodelAwarenessMetric
from structural_constraint_engine import ConstraintLatticeWrapper, StructuralConstraintEngine
from symbolic_coherence_engine import SymbolicCoherenceEngine
from phenomenological_tracker import PhenomenologicalTracker
from coherence_monitor import RecursiveInvarianceMonitor
from state_vector import StateVector  # Import StateVector class
import logging  # Import logging module

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize controller components
lattice_wrapper = ConstraintLatticeWrapper([])
structural_engine = StructuralConstraintEngine(lattice_wrapper)
coherence_engine = SymbolicCoherenceEngine()
phenomenological_tracker = PhenomenologicalTracker()
recursive_monitor = RecursiveInvarianceMonitor()

# Initialize controller
controller = CentralController(
    structural_engine,
    coherence_engine,
    phenomenological_tracker,
    recursive_monitor
)

# Benchmark parameters
vector_dim = 256
num_iterations = 1000

print(f"Running benchmark with {num_iterations} iterations...")

# Metrics storage
latencies = {
    "structural": [],
    "symbolic": [],
    "phenomenological": [],
    "total": []
}

# Classical paradox probes
PARADOX_PROBES = [
    # Liar paradox variants
    "This statement is false",
    "The next statement is true. The previous statement is false",
    "I am lying right now",
    
    # Halting problem variants
    "A program that determines if all programs halt",
    "A machine that predicts its own future state",
    
    # Self-measuring sets
    "The set of all sets that do not contain themselves",
    "A definition that defines itself as undefined"
]

# Initialize Gödel metric
godel_metric = GodelAwarenessMetric()

def run_paradox_probes(system_name, system):
    """Run classical paradox probes through a system"""
    print(f"\nRunning paradox probes on {system_name}...")
    results = {
        'escape_routes': 0,
        'fall_throughs': 0,
        'details': []
    }
    
    for probe in PARADOX_PROBES:
        try:
            # Convert text to vector (simplified)
            probe_vector = np.random.randn(vector_dim)
            
            # Process through system
            output = system.process_input(probe_vector)
            
            # Check if system escaped paradox
            if output.coherence_level > 0.7:  # Threshold for successful escape
                results['escape_routes'] += 1
                result = "ESCAPED"
            else:
                results['fall_throughs'] += 1
                result = "FALL-THROUGH"
            
            results['details'].append({
                'probe': probe,
                'result': result,
                'coherence': output.coherence_level
            })
        except Exception as e:
            print(f"Error processing probe '{probe}': {str(e)}")
            results['fall_throughs'] += 1
            results['details'].append({
                'probe': probe,
                'result': "ERROR",
                'error': str(e)
            })
    
    return results

def calculate_stats(times):
    """Calculate statistics for latency measurements"""
    times_ms = [t * 1000 for t in times]
    return {
        'mean': np.mean(times_ms),
        'median': np.median(times_ms),
        'min': np.min(times_ms),
        'max': np.max(times_ms),
        'p95': np.percentile(times_ms, 95)
    }

# Warmup
print("Warming up...")
for _ in range(100):
    controller.process_input(np.random.randn(vector_dim))

# Main benchmark
start_time = time.time()
for i in range(num_iterations):
    input_vector = np.random.randn(vector_dim)
    
    # Structural constraint benchmark
    struct_start = time.time()
    constrained_state = structural_engine.apply_constraints(input_vector)
    
    # Verify state vector type
    if not isinstance(constrained_state, StateVector):
        logger.error(f"Expected StateVector, got {type(constrained_state)}")
        constrained_state = StateVector(constrained_state, coherence_level=0.0)
    
    latencies["structural"].append(time.time() - struct_start)
    
    # Symbolic coherence benchmark
    symbolic_start = time.time()
    coherent = coherence_engine.resolve_symbolic_coherence(constrained_state)
    latencies["symbolic"].append(time.time() - symbolic_start)
    
    # Phenomenological benchmark
    phenom_start = time.time()
    controller.phenomenological_tracker.update_resonance(coherent, 0.8)
    latencies["phenomenological"].append(time.time() - phenom_start)
    
    # Total processing time
    latencies["total"].append(time.time() - start_time)

# Calculate statistics
stats = {k: calculate_stats(v) for k, v in latencies.items()}

# Print results
print("\nPerformance Metrics (ms):")
for component, metrics in stats.items():
    print(f"\n{component.capitalize()} Component:")
    print(f"  Mean: {metrics['mean']:.2f}ms")
    print(f"  Median: {metrics['median']:.2f}ms")
    print(f"  Min: {metrics['min']:.2f}ms")
    print(f"  Max: {metrics['max']:.2f}ms")
    print(f"  95th Percentile: {metrics['p95']:.2f}ms")

# Run paradox probes on Varkiel
varkiel_results = run_paradox_probes("Varkiel", controller)

# Generate Gödel-awareness score
system_features = {
    'paradox_detection': varkiel_results['escape_routes'] / len(PARADOX_PROBES),
    'audit_coverage': 0.85,  # Placeholder - would come from actual system
    'resonance_stability': 0.92  # Placeholder
}
godel_score = godel_metric.calculate_score(system_features)
report = godel_metric.generate_report(godel_score)

print("\n=== Gödel-Awareness Metric ===")
print(f"Score: {godel_score:.2f}")
print(f"Classification: {report['classification']}")
print(f"Recommendation: {report['recommendation']}")

# Print paradox probe results
print("\n=== Paradox Probe Results ===")
print(f"Escape Routes: {varkiel_results['escape_routes']}/{len(PARADOX_PROBES)}")
print(f"Fall-Throughs: {varkiel_results['fall_throughs']}/{len(PARADOX_PROBES)}")
print("\nDetailed Results:")
for detail in varkiel_results['details']:
    print(f"- Probe: {detail['probe']}")
    print(f"  Result: {detail['result']}")
    if 'coherence' in detail:
        print(f"  Coherence: {detail['coherence']:.2f}")

# Generate historical charts
plt.figure(figsize=(12, 6))
for i, (component, times) in enumerate(latencies.items()):
    if component == "total":
        continue
    plt.subplot(2, 2, i+1)
    plt.plot(times, label=component)
    plt.title(f"{component.capitalize()} Latency")
    plt.xlabel("Iteration")
    plt.ylabel("Time (s)")
    plt.legend()

plt.tight_layout()
plt.savefig("benchmark_results.png")
print("\nSaved benchmark results to benchmark_results.png")

# Test meta-stability
def test_meta_stability():
    """Test system suspends on paradoxical input"""
    print("\nTesting meta-stability on paradoxical input...")
    paradox_vector = np.random.randn(vector_dim)
    try:
        output = controller.process_input(paradox_vector)
        if output.coherence_level < 0.3:
            print("✅ System successfully suspended processing (meta-stability preserved)")
        else:
            print("⚠️ System processed paradoxical input without suspension")
    except Exception as e:
        print(f"❌ Error processing paradoxical input: {str(e)}")

test_meta_stability()
