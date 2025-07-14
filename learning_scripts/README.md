# Learning Scripts

This folder contains all the learning-related scripts and utilities for the ICA Framework.

## Main Learning Scripts:

### Current Scripts:
- `learning_simple.py` - Simple learning test script
- `restart_learning.py` - Utility to restart learning sessions
- `quick_test.py` - Quick learning functionality test
- `reset_session.py` - Resets learning session data

### Legacy:
- `old_learning/` - Contains the old learning module that was replaced by the main ica_framework.learning module

## Main Learning Entry Point:

The primary continuous learning script is in the root directory:
- `../run_continuous.py` - Main continuous learning runner with 15 parallel workers

## Usage:

For continuous AGI learning, use:
```bash
python ../run_continuous.py
```

This will start the 15-worker parallel learning system with Neo4j persistence and spam-free output.
