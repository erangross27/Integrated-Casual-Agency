#!/usr/bin/env python3
"""
Quick test to verify ContinuousLearning methods
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ica_framework.learning import ContinuousLearning

# Check available methods
learning = ContinuousLearning(
    database_backend="memory",
    num_workers=1,
    enable_parallel=False
)

print("Available methods in ContinuousLearning:")
methods = [method for method in dir(learning) if not method.startswith('_') and callable(getattr(learning, method))]
for method in methods:
    print(f"  • {method}")

print(f"\nHas 'run' method: {'run' in methods}")
print(f"Has 'run_continuous_learning' method: {'run_continuous_learning' in methods}")
