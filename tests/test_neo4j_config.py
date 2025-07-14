#!/usr/bin/env python3
"""
Test Neo4j Configuration Loading
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_config_loading():
    """Test loading Neo4j configuration"""
    print("🔍 Testing Neo4j configuration loading...")
    
    config_file = Path("config/database/neo4j.json")
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            db_config = config_data['config']
            print("✅ Neo4j configuration loaded successfully:")
            print(f"   URI: {db_config['uri']}")
            print(f"   Database: {db_config['database']}")
            print(f"   Username: {db_config['username']}")
            print(f"   Password: {'*' * len(db_config['password'])}")
            
            # Test with ContinuousLearning
            from ica_framework.learning import ContinuousLearning
            
            learning = ContinuousLearning(
                database_backend="neo4j",
                database_config=db_config,
                num_workers=1,
                enable_parallel=False
            )
            
            print("✅ ContinuousLearning created with proper config")
            return True
            
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return False
    else:
        print(f"❌ Config file not found: {config_file}")
        return False

if __name__ == "__main__":
    test_config_loading()
