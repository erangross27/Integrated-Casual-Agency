#!/usr/bin/env python3
"""
Causal Model Storage Module
Handles storage and retrieval of AGI causal models
"""

import json
import time


class CausalModelStorage:
    """Handles AGI causal model storage operations"""
    
    def __init__(self, knowledge_graph, session_id):
        self.kg = knowledge_graph
        self.session_id = session_id
        self.models_stored = 0
    
    def store_causal_models(self, causal_models):
        """Store causal models"""
        if not causal_models or len(causal_models) == 0:
            return 0
        
        stored_count = 0
        
        for i, model in enumerate(causal_models):
            model_id = f"agi_causal_model_{i}_{int(time.time() * 1000)}_{self.session_id}"
            model_entity = {
                'type': 'agi_causal_model',
                'model_data': json.dumps(model) if isinstance(model, dict) else str(model),
                'created_at': time.time(),
                'session_id': self.session_id,
                'source': 'agi_learning'
            }
            
            if self.kg.add_entity(model_id, 'agi_causal_model', model_entity):
                stored_count += 1
        
        self.models_stored += stored_count
        return stored_count
    
    def get_stored_models(self, session_id=None):
        """Get stored causal models from database"""
        if not self.kg:
            return []
        
        try:
            if session_id:
                query = f"MATCH (n:Entity {{type: 'agi_causal_model', session_id: '{session_id}'}}) RETURN n ORDER BY n.created_at DESC"
            else:
                query = f"MATCH (n:Entity {{type: 'agi_causal_model'}}) RETURN n ORDER BY n.created_at DESC"
            
            result = self.kg.execute_custom_query(query)
            
            models = []
            for record in result:
                model_data = record.get('n', {})
                
                # Parse model data
                raw_model_data = model_data.get('model_data', '{}')
                try:
                    parsed_model = json.loads(raw_model_data)
                except:
                    parsed_model = raw_model_data
                
                models.append({
                    'id': model_data.get('id', 'unknown'),
                    'model_data': parsed_model,
                    'created_at': model_data.get('created_at', 0),
                    'session_id': model_data.get('session_id', 'unknown')
                })
            
            return models
            
        except Exception as e:
            print(f"[CAUSAL] ⚠️ Causal model retrieval error: {e}")
            return []
    
    def restore_models_to_agent(self, agi_agent):
        """Restore causal models from database to AGI agent"""
        if not self.kg or not agi_agent:
            return False
        
        try:
            # Get all stored models from any session (for restoration)
            stored_models = self.get_stored_models()  # No session_id = get all
            
            if not stored_models:
                return False
            
            # Restore to agent if it has causal model storage
            if hasattr(agi_agent, 'causal_models'):
                if not hasattr(agi_agent.causal_models, 'clear'):
                    agi_agent.causal_models = []
                
                for model in stored_models:
                    if model['model_data'] not in agi_agent.causal_models:
                        agi_agent.causal_models.append(model['model_data'])
            
            print(f"💾 [CAUSAL] ✅ Restored {len(stored_models)} causal models to agent")
            return True
            
        except Exception as e:
            print(f"[CAUSAL] ⚠️ Causal model restoration error: {e}")
            return False
    
    def get_stats(self):
        """Get causal model storage statistics"""
        return {
            'models_stored': self.models_stored
        }
