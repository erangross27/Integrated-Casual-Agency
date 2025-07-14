# Test Scripts

This folder contains all the debugging and testing scripts created during the ICA Framework development.

## Key Test Files:

### Worker Testing:
- `test_actual_worker.py` - Tests the real continuous_parallel_worker function
- `test_worker_simple.py` - Simple worker test without early stop events
- `test_worker_results.py` - Tests worker result collection
- `test_exact_worker_flow.py` - Tests exact worker workflow
- `test_multiprocessing_neo4j.py` - Tests multiprocessing with Neo4j

### Scenario Testing:
- `test_scenario_generation.py` - Tests enhanced scenario generation
- `test_simple_multiprocessing.py` - Basic multiprocessing functionality test

### Database Testing:
- `test_neo4j_config.py` - Tests Neo4j configuration
- `test_entity_creation.py` - Tests entity creation in Neo4j
- `test_observe_environment.py` - Tests observation processing

### Component Testing:
- `test_methods.py` - Tests various ICA Framework methods
- `test_modular.py` - Tests modular components

### Debug Scripts:
- `debug_worker.py` - Worker debugging utilities

All these tests were crucial for identifying and fixing the continuous learning issues.
