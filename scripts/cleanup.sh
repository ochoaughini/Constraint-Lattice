#!/bin/bash

# Remove duplicate and backup files
rm .dockerignore.bak
rm .dockerignore.bak.bak
rm .env.bak

# Remove deprecated modules
rm -r adapters
rm -r billing
rm -r clattice
rm -r deployment
rm -r hf-cache
rm -r infra
rm -r kubernetes
rm -r minimal_test
rm -r pipelines
rm -r saas
rm -r services
rm -r tools
rm -r wp-plugin

# Remove unused test files
rm tests/unit/test_core_no_saas_flag.py
rm tests/unit/test_dual_gen.py
rm tests/unit/test_dual_mode.py
rm tests/unit/test_jax_backend.py
rm tests/unit/test_jax_constraints_extended.py
rm tests/unit/test_kafka_sink.py
rm tests/unit/test_loader_jax_integration.py
rm tests/unit/test_saas_flag_enabled.py
rm tests/unit/test_sdk_engine.py
rm tests/unit/test_sdk_init.py

# Remove experimental code
rm multimodal.py
rm policy_dsl.py
rm constraint_test.py
