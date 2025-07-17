#!/usr/bin/env python3
"""
Neural Network Persistence Verification Tool
Test if neural networks are properly saved and can be restored
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.components.database.postgresql_agi_persistence import PostgreSQLAGIPersistence


def verify_neural_models_in_db():
    """Check what neural models are stored in the database"""
    print("🔍 Checking neural models in PostgreSQL database...")
    
    # Use current session (if available) or create test session
    session_id = f"verification_session_{int(time.time())}"
    agi_persistence = PostgreSQLAGIPersistence(session_id)
    
    try:
        with agi_persistence.connection.cursor() as cursor:
            # Check neural models table
            cursor.execute("""
                SELECT session_id, model_name, version, created_at, 
                       model_size_bytes, parameter_count, is_current
                FROM neural_models 
                ORDER BY created_at DESC
                LIMIT 10
            """)
            
            models = cursor.fetchall()
            
            if models:
                print(f"✅ Found {len(models)} neural model records:")
                print("=" * 80)
                for model in models:
                    session_id, name, version, created, size_bytes, params, current = model
                    size_mb = size_bytes / (1024 * 1024) if size_bytes else 0
                    status = "🟢 CURRENT" if current else "🔶 OLD"
                    print(f"📊 {name} v{version} - {size_mb:.1f}MB ({params:,} params) {status}")
                    print(f"   Session: {session_id}")
                    print(f"   Created: {created}")
                    print()
                
                # Check total storage
                cursor.execute("""
                    SELECT COUNT(*), SUM(model_size_bytes) 
                    FROM neural_models 
                    WHERE is_current = TRUE
                """)
                
                count, total_size = cursor.fetchone()
                total_mb = total_size / (1024 * 1024) if total_size else 0
                print(f"💾 Total: {count} current models, {total_mb:.1f}MB storage used")
                
            else:
                print("ℹ️ No neural models found in database")
                
            # Check sessions
            cursor.execute("""
                SELECT session_id, started_at, environment_type
                FROM agi_sessions 
                ORDER BY started_at DESC
                LIMIT 5
            """)
            
            sessions = cursor.fetchall()
            if sessions:
                print("\n🔄 Recent AGI Sessions:")
                for session in sessions:
                    print(f"   • {session[0]} - {session[1]} ({session[2] or 'unknown env'})")
            
            return len(models) > 0
            
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False


def test_neural_save_restore():
    """Test neural network save and restore functionality"""
    print("\n🧪 Testing neural network save/restore...")
    
    try:
        # Create a simple test model
        test_model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # Save original weights
        original_weights = test_model.state_dict().copy()
        
        # Initialize persistence
        import time
        session_id = f"test_session_{int(time.time())}"
        agi_persistence = PostgreSQLAGIPersistence(session_id)
        
        print(f"💾 Saving test model to session: {session_id}")
        
        # Save the model
        save_success = agi_persistence.save_neural_model(
            'test_model', 
            test_model,
            {'test': True, 'verification': True}
        )
        
        if not save_success:
            print("❌ Failed to save test model")
            return False
        
        # Modify the model (simulate learning)
        with torch.no_grad():
            for param in test_model.parameters():
                param.fill_(0.999)  # Fill with different values
        
        print("🔄 Loading test model from database...")
        
        # Restore the model
        load_success = agi_persistence.load_neural_model('test_model', test_model)
        
        if not load_success:
            print("❌ Failed to load test model")
            return False
        
        # Compare weights
        restored_weights = test_model.state_dict()
        weights_match = True
        
        for key in original_weights.keys():
            if not torch.allclose(original_weights[key], restored_weights[key], atol=1e-6):
                weights_match = False
                break
        
        if weights_match:
            print("✅ Neural network save/restore test PASSED!")
            print("🧠 Weights successfully persisted and restored")
            return True
        else:
            print("❌ Neural network save/restore test FAILED!")
            print("⚠️ Restored weights don't match original")
            return False
            
    except Exception as e:
        print(f"❌ Save/restore test failed: {e}")
        return False


def main():
    """Main verification function"""
    print("🔍 Neural Network Persistence Verification")
    print("=" * 50)
    
    # Check what's in the database
    models_exist = verify_neural_models_in_db()
    
    # Test save/restore functionality
    test_passed = test_neural_save_restore()
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY:")
    print(f"🗄️ Models in Database: {'✅ YES' if models_exist else '❌ NO'}")
    print(f"🔄 Save/Restore Test: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    if models_exist and test_passed:
        print("🎉 Neural persistence is working correctly!")
        print("💡 Your AGI knowledge is being properly saved and can be restored")
    else:
        print("⚠️ Issues detected with neural persistence")
        print("💡 AGI may not be retaining learned knowledge between sessions")


if __name__ == "__main__":
    import time
    main()
