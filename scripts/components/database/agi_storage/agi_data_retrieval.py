#!/usr/bin/env python3
"""
AGI Data Retrieval Module
Handles data extraction from AGI agents
"""


class AGIDataRetrieval:
    """Handles data extraction from AGI agents"""
    
    def __init__(self):
        pass
    
    def extract_learning_data(self, agi_agent):
        """Extract all learning data from AGI agent"""
        if not agi_agent:
            return {
                'knowledge_base': [],
                'active_hypotheses': [],
                'confirmed_hypotheses': [],
                'causal_models': []
            }
        
        try:
            # Get AGI's actual learning data
            knowledge_base = agi_agent.get_knowledge_base() if hasattr(agi_agent, 'get_knowledge_base') else []
            active_hypotheses = agi_agent.get_active_hypotheses() if hasattr(agi_agent, 'get_active_hypotheses') else []
            confirmed_hypotheses = agi_agent.get_confirmed_hypotheses() if hasattr(agi_agent, 'get_confirmed_hypotheses') else []
            causal_models = agi_agent.get_causal_models() if hasattr(agi_agent, 'get_causal_models') else []
            
            return {
                'knowledge_base': knowledge_base,
                'active_hypotheses': active_hypotheses,
                'confirmed_hypotheses': confirmed_hypotheses,
                'causal_models': causal_models
            }
            
        except Exception as e:
            print(f"[DATA] ⚠️ Data extraction error: {e}")
            return {
                'knowledge_base': [],
                'active_hypotheses': [],
                'confirmed_hypotheses': [],
                'causal_models': []
            }
    
    def count_learning_data(self, learning_data):
        """Count total learning data items"""
        return (
            len(learning_data.get('knowledge_base', [])) +
            len(learning_data.get('active_hypotheses', [])) +
            len(learning_data.get('confirmed_hypotheses', [])) +
            len(learning_data.get('causal_models', []))
        )
    
    def validate_learning_data(self, learning_data):
        """Validate learning data structure"""
        required_keys = ['knowledge_base', 'active_hypotheses', 'confirmed_hypotheses', 'causal_models']
        
        for key in required_keys:
            if key not in learning_data:
                return False
            
            if not isinstance(learning_data[key], list):
                return False
        
        return True
    
    def get_learning_summary(self, learning_data):
        """Get summary of learning data"""
        if not self.validate_learning_data(learning_data):
            return "Invalid learning data"
        
        summary = {
            'concepts': len(learning_data['knowledge_base']),
            'active_hypotheses': len(learning_data['active_hypotheses']),
            'confirmed_hypotheses': len(learning_data['confirmed_hypotheses']),
            'causal_models': len(learning_data['causal_models']),
            'total_items': self.count_learning_data(learning_data)
        }
        
        return summary
