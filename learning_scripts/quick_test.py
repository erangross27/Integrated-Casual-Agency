#!/usr/bin/env python3
"""
Quick Test - ICA Framework Modular System
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print("🔍 Testing ICA Framework Modular Learning...")

try:
    # Test basic imports
    from ica_framework.learning import ContinuousLearning
    print("✅ ContinuousLearning imported")
    
    from ica_framework.learning import ComprehensiveScenarioLibrary
    print("✅ ComprehensiveScenarioLibrary imported")
    
    # Test scenario generation
    lib = ComprehensiveScenarioLibrary()
    scenarios = lib.create_iot_scenarios()
    print(f"✅ Generated {len(scenarios)} IoT scenarios")
    
    # Test learning instance creation
    learning = ContinuousLearning(
        database_backend="memory",
        num_workers=2,
        enable_parallel=True,
        batch_size=5
    )
    print("✅ ContinuousLearning instance created")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("Your modular system is working correctly!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
