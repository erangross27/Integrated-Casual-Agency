#!/usr/bin/env python3
"""
Test Continuous Worker Entity Creation
Test if the continuous learning workers are actually creating entities with the new format
"""

import sys
from pathlib import Path
import time
import multiprocessing as mp
from queue import Queue
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ica_framework.learning.worker_functions import continuous_parallel_worker, _create_observation
from ica_framework.learning.scenario_generators import PhysicsSimulation
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

def load_database_config():
    """Load Neo4j configuration"""
    config_file = Path("config/database/neo4j.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            return config_data['config']
    return {
        'uri': 'neo4j://127.0.0.1:7687',
        'username': 'neo4j',
        'password': 'password',
        'database': 'neo4j'
    }

def test_worker_scenario_creation():
    """Test if worker scenario creation generates proper entities"""
    print("🧪 Testing worker scenario creation...")
    
    # Create physics simulation like workers do
    physics_sim = PhysicsSimulation()
    
    # Generate a test scenario
    scenario = physics_sim.generate_physics_scenario(complexity_level=0.5)
    print(f"📋 Generated scenario keys: {list(scenario.keys())}")
    scenario_name = scenario.get('name', 'physics_scenario')
    print(f"📋 Generated scenario: {scenario_name}")
    print(f"📊 Raw entities: {scenario.get('entities', [])}")
    print(f"📊 Raw relationships: {len(scenario.get('relationships', []))} relationships")
    
    # Test the _create_observation function
    observation = _create_observation(scenario, current_round=1)
    
    print(f"\n🔍 Observation structure:")
    print(f"   Entities: {len(observation['entities'])}")
    print(f"   Relationships: {len(observation['relationships'])}")
    print(f"   State shape: {observation['state'].shape}")
    
    # Check if entities have proper format
    if observation['entities']:
        first_entity = observation['entities'][0]
        print(f"\n📝 First entity format:")
        print(f"   ID: {first_entity.get('id', 'MISSING')}")
        print(f"   Label: {first_entity.get('label', 'MISSING')}")
        print(f"   Properties static: {first_entity.get('properties_static', 'MISSING')}")
        print(f"   Properties dynamic: {first_entity.get('properties_dynamic', 'MISSING')}")
        print(f"   Confidence: {first_entity.get('confidence', 'MISSING')}")
        
        if all(key in first_entity for key in ['id', 'label', 'properties_static', 'properties_dynamic', 'confidence']):
            print("✅ Entity format is CORRECT")
        else:
            print("❌ Entity format is MISSING REQUIRED FIELDS")
    
    # Check relationships
    if observation['relationships']:
        first_rel = observation['relationships'][0]
        print(f"\n🔗 First relationship format:")
        print(f"   Source: {first_rel.get('source', 'MISSING')}")
        print(f"   Target: {first_rel.get('target', 'MISSING')}")
        print(f"   Type: {first_rel.get('type', 'MISSING')}")
        print(f"   Confidence: {first_rel.get('confidence', 'MISSING')}")

def test_single_worker_execution():
    """Test a single worker execution"""
    print("\n🔧 Testing single worker execution...")
    
    database_config = load_database_config()
    
    # Create queues for worker communication
    scenario_queue = mp.Queue()
    results_queue = mp.Queue()
    stop_event = mp.Event()
    
    # Add a test scenario to the queue
    physics_sim = PhysicsSimulation()
    scenario = physics_sim.generate_physics_scenario(complexity_level=0.5)
    scenario_info = (1, scenario, 1)  # scenario_count, scenario, current_round
    scenario_queue.put(scenario_info)
    
    # Get initial node count
    enhanced_kg = EnhancedKnowledgeGraph(backend="neo4j", config=database_config)
    if enhanced_kg.connect():
        initial_stats = enhanced_kg.get_stats()
        print(f"📊 Initial stats: {initial_stats['nodes']} nodes, {initial_stats['edges']} edges")
        enhanced_kg.disconnect()
    
    print("🚀 Starting single worker...")
    
    # Start worker for just one scenario
    worker_process = mp.Process(
        target=continuous_parallel_worker,
        args=(0, scenario_queue, results_queue, stop_event, database_config, "neo4j")
    )
    worker_process.start()
    
    # Wait a bit for processing
    time.sleep(2)
    
    # Stop the worker
    stop_event.set()
    worker_process.join(timeout=5)
    if worker_process.is_alive():
        worker_process.terminate()
    
    # Check results
    results = []
    while not results_queue.empty():
        try:
            result = results_queue.get_nowait()
            results.append(result)
        except:
            break
    
    print(f"📊 Worker results: {len(results)} results")
    for result in results:
        if 'worker_id' in result:
            print(f"   Worker {result['worker_id']}: +{result.get('nodes_added', 0)} nodes, +{result.get('edges_added', 0)} edges")
    
    # Check final node count
    enhanced_kg = EnhancedKnowledgeGraph(backend="neo4j", config=database_config)
    if enhanced_kg.connect():
        final_stats = enhanced_kg.get_stats()
        print(f"📊 Final stats: {final_stats['nodes']} nodes, {final_stats['edges']} edges")
        
        nodes_added = final_stats['nodes'] - initial_stats['nodes']
        edges_added = final_stats['edges'] - initial_stats['edges']
        
        print(f"➕ Total added: {nodes_added} nodes, {edges_added} edges")
        
        if nodes_added > 0:
            print("✅ SUCCESS: Worker created entities!")
        else:
            print("❌ PROBLEM: Worker did not create entities!")
        
        enhanced_kg.disconnect()

if __name__ == "__main__":
    test_worker_scenario_creation()
    test_single_worker_execution()
