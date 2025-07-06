# ADR 0002: Instrumentation and ADR Enforcement

## Status
Accepted

## Context
We need to ensure architectural decisions are documented and system behavior is observable. This ADR introduces:
1. Pre-commit hook for ADR enforcement when core logic changes
2. Prometheus instrumentation for critical performance metrics

## Decision
- Created pre-commit hook that requires ADR entries for core changes
- Added Prometheus metrics for MetaConstraintTree cache
- Exposed metrics endpoint on port 8000

## Consequences
- Improved architectural visibility
- Real-time performance monitoring
- Additional dependency on Prometheus client
