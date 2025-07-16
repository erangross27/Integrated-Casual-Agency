#!/usr/bin/env python3
"""
Neural Network Persistence Module
Handles saving and loading of neural network weights and states
"""

import torch
import pickle
import base64
import json
import time


class NeuralPersistence:
    """Handles neural network model persistence"""
    
    def __init__(self, knowledge_graph, session_id):
        self.kg = knowledge_graph
        self.session_id = session_id
        self.models_stored = 0
    
    def save_model_weights(self, model_name, model_state_dict, metadata=None):
        """Save neural network model weights to database"""
        if not self.kg:
            print(f"💾 [NEURAL] ⚠️ No knowledge graph connection for {model_name}")
            return False
        
        try:
            # Serialize model state dict
            model_bytes = pickle.dumps(model_state_dict)
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            
            model_entity = {
                'type': 'neural_model',
                'model_name': model_name,
                'session_id': self.session_id,
                'model_weights': model_b64,
                'metadata': json.dumps(metadata or {}),
                'timestamp': time.time(),
                'is_current': True
            }
            
            # Mark previous models as not current
            if hasattr(self.kg, 'execute_custom_query'):
                try:
                    self.kg.execute_custom_query(
                        f"MATCH (n:Entity {{type: 'neural_model', model_name: '{model_name}'}}) SET n.is_current = false"
                    )
                except Exception as e:
                    print(f"💾 [NEURAL] ⚠️ Warning: Could not mark previous models as not current: {e}")
            
            model_id = f"model_{model_name}_{self.session_id}_{int(time.time())}"
            if self.kg.add_entity(model_id, 'neural_model', model_entity):
                self.models_stored += 1
                print(f"💾 [NEURAL] ✅ Saved {model_name} weights ({len(model_bytes)} bytes)")
                return True
            else:
                print(f"💾 [NEURAL] ⚠️ Failed to add {model_name} entity to database")
                return False
            
        except Exception as e:
            print(f"💾 [NEURAL] ⚠️ Model save error for {model_name}: {e}")
            return False
    
    def load_model_weights(self, model_name, session_id=None):
        """Load neural network model weights from database"""
        if not self.kg:
            return None
        
        try:
            # Query for latest model
            if session_id:
                query = f"MATCH (n:Entity {{type: 'neural_model', model_name: '{model_name}', session_id: '{session_id}'}}) RETURN n ORDER BY n.timestamp DESC LIMIT 1"
            else:
                query = f"MATCH (n:Entity {{type: 'neural_model', model_name: '{model_name}', is_current: true}}) RETURN n ORDER BY n.timestamp DESC LIMIT 1"
            
            result = self.kg.execute_custom_query(query)
            if result and len(result) > 0:
                model_data = result[0].get('n', {})
                model_b64 = model_data.get('model_weights', '')
                
                if model_b64:
                    model_bytes = base64.b64decode(model_b64)
                    model_state_dict = pickle.loads(model_bytes)
                    
                    metadata = json.loads(model_data.get('metadata', '{}'))
                    print(f"💾 [NEURAL] Loaded {model_name} model weights ({len(model_bytes)} bytes)")
                    
                    return {
                        'state_dict': model_state_dict,
                        'metadata': metadata,
                        'timestamp': model_data.get('timestamp', 0)
                    }
            
        except Exception as e:
            print(f"[NEURAL] ⚠️ Model load error: {e}")
        
        return None
    
    def save_gpu_models(self, gpu_processor):
        """Save all GPU model weights"""
        if not gpu_processor:
            print("💾 [NEURAL] No GPU processor provided")
            return False
            
        if not gpu_processor.use_gpu:
            print("💾 [NEURAL] GPU not enabled")
            return False
        
        saved_count = 0
        
        # Save pattern recognizer
        if hasattr(gpu_processor, 'pattern_recognizer') and gpu_processor.pattern_recognizer:
            try:
                if self.save_model_weights(
                    'pattern_recognizer', 
                    gpu_processor.pattern_recognizer.state_dict(),
                    {'input_size': getattr(gpu_processor.pattern_recognizer, 'input_size', 'unknown')}
                ):
                    saved_count += 1
            except Exception as e:
                print(f"💾 [NEURAL] ⚠️ Error saving pattern_recognizer: {e}")
        
        # Save hypothesis generator
        if hasattr(gpu_processor, 'hypothesis_generator') and gpu_processor.hypothesis_generator:
            try:
                if self.save_model_weights(
                    'hypothesis_generator', 
                    gpu_processor.hypothesis_generator.state_dict(),
                    {'input_size': getattr(gpu_processor.hypothesis_generator, 'input_size', 'unknown')}
                ):
                    saved_count += 1
            except Exception as e:
                print(f"💾 [NEURAL] ⚠️ Error saving hypothesis_generator: {e}")
        
        if saved_count > 0:
            print(f"💾 [NEURAL] ✅ Saved {saved_count} GPU models to Neo4j")
        else:
            print(f"💾 [NEURAL] ⚠️ No GPU models saved - check model initialization")
        
        return saved_count > 0
    
    def restore_gpu_models(self, gpu_processor):
        """Restore all GPU model weights"""
        if not gpu_processor or not gpu_processor.use_gpu:
            return False
        
        restored_count = 0
        
        # Restore pattern recognizer
        pattern_weights = self.load_model_weights('pattern_recognizer')
        if pattern_weights and hasattr(gpu_processor, 'pattern_recognizer') and gpu_processor.pattern_recognizer:
            gpu_processor.pattern_recognizer.load_state_dict(pattern_weights['state_dict'])
            print(f"💾 [NEURAL] Restored pattern recognizer weights")
            restored_count += 1
        
        # Restore hypothesis generator
        hypothesis_weights = self.load_model_weights('hypothesis_generator')
        if hypothesis_weights and hasattr(gpu_processor, 'hypothesis_generator') and gpu_processor.hypothesis_generator:
            gpu_processor.hypothesis_generator.load_state_dict(hypothesis_weights['state_dict'])
            print(f"💾 [NEURAL] Restored hypothesis generator weights")
            restored_count += 1
        
        print(f"💾 [NEURAL] Restored {restored_count} GPU models")
        return restored_count > 0
    
    def get_stats(self):
        """Get neural persistence statistics"""
        return {
            'models_stored': self.models_stored
        }
