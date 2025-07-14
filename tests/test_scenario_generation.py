#!/usr/bin/env python3
"""
Test scenario generation to see if scenarios actually contain entities
"""

import time
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_scenario_generation():
    """Test what scenarios are actually being generated"""
    
    print("🧪 Testing scenario generation with FIXED worker functions...")
    
    # Import scenario generators AND the fixed worker function
    from ica_framework.learning.scenario_generators import PhysicsSimulation
    from ica_framework.sandbox import ProceduralDatasetGenerator, MultiDomainScenarioGenerator
    from ica_framework.utils.config import SandboxConfig
    from ica_framework.learning.worker_functions import _generate_enhanced_scenario
    
    # Initialize generators like in worker_functions.py
    physics_sim = PhysicsSimulation()
    
    try:
        sandbox_config = SandboxConfig()
        procedural_gen = ProceduralDatasetGenerator(sandbox_config)
        multi_domain_gen = MultiDomainScenarioGenerator(sandbox_config)
    except Exception as e:
        print(f"⚠️ Could not create enhanced generators: {e}")
        procedural_gen = None
        multi_domain_gen = None
    
    # Test all scenario types using the FIXED _generate_enhanced_scenario function
    base_scenario = {}  # Empty base scenario
    
    print("\n📊 Testing different scenario types with FIXED function:")
    
    for scenario_type in range(10):
        scenario_count = 100 + scenario_type  # Simulate scenario count
        current_round = 1
        
        print(f"\n🎯 Scenario Type {scenario_type}:")
        
        # Use the ACTUAL fixed function from worker_functions.py
        scenario = _generate_enhanced_scenario(
            scenario_type, 
            scenario_count, 
            current_round,
            physics_sim, 
            procedural_gen, 
            multi_domain_gen, 
            base_scenario
        )
        
        # Show final scenario content
        entities_count = len(scenario.get('entities', []))
        relationships_count = len(scenario.get('relationships', []))
        
        if entities_count == 0:
            print(f"   🚨 STILL BROKEN: Scenario type {scenario_type} has NO ENTITIES!")
        else:
            print(f"   ✅ FIXED: {entities_count} entities, {relationships_count} relationships")
            print(f"   📝 Name: {scenario.get('name', 'Unknown')}")
    
    print(f"\n📊 Summary:")
    print(f"All scenario types should now have entities thanks to the fallback function!")

if __name__ == "__main__":
    test_scenario_generation()
