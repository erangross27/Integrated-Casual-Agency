#!/usr/bin/env python3
"""
AGI Learning Coordinator Module
Coordinates all AGI learning storage operations
"""

from .concept_storage import ConceptStorage
from .hypothesis_storage import HypothesisStorage
from .causal_model_storage import CausalModelStorage
from .agi_data_retrieval import AGIDataRetrieval


class AGILearningCoordinator:
    """Coordinates all AGI learning storage operations"""
    
    def __init__(self, knowledge_graph, session_id):
        self.kg = knowledge_graph
        self.session_id = session_id
        
        # Initialize storage components
        self.concept_storage = ConceptStorage(knowledge_graph, session_id)
        self.hypothesis_storage = HypothesisStorage(knowledge_graph, session_id)
        self.causal_storage = CausalModelStorage(knowledge_graph, session_id)
        self.data_retrieval = AGIDataRetrieval()
    
    def store_agi_learning(self, agi_agent):
        """Store complete AGI learning data"""
        if not agi_agent or not self.kg:
            return False
        
        try:
            # Extract learning data from agent
            learning_data = self.data_retrieval.extract_learning_data(agi_agent)
            
            if not self.data_retrieval.validate_learning_data(learning_data):
                return False
            
            # Check if there's data to store
            if self.data_retrieval.count_learning_data(learning_data) == 0:
                return True
            
            # Store each type of data
            concepts_stored = self.concept_storage.store_concepts(learning_data['knowledge_base'])
            active_hyp_stored = self.hypothesis_storage.store_active_hypotheses(learning_data['active_hypotheses'])
            confirmed_hyp_stored = self.hypothesis_storage.store_confirmed_hypotheses(learning_data['confirmed_hypotheses'])
            models_stored = self.causal_storage.store_causal_models(learning_data['causal_models'])
            
            # Return success if any data was stored
            return (concepts_stored > 0 or active_hyp_stored > 0 or 
                   confirmed_hyp_stored > 0 or models_stored > 0)
            
        except Exception as e:
            print(f"[COORDINATOR] ⚠️ Storage error: {e}")
            return False
    
    def restore_agi_learning(self, agi_agent):
        """Restore complete AGI learning data"""
        if not agi_agent or not self.kg:
            return False
        
        try:
            # Restore each type of data
            concepts_restored = self.concept_storage.restore_concepts_to_agent(agi_agent)
            hypotheses_restored = self.hypothesis_storage.restore_hypotheses_to_agent(agi_agent)
            models_restored = self.causal_storage.restore_models_to_agent(agi_agent)
            
            return concepts_restored or hypotheses_restored or models_restored
            
        except Exception as e:
            print(f"[COORDINATOR] ⚠️ Restoration error: {e}")
            return False
    
    def get_stored_concepts(self, session_id=None):
        """Get stored concepts"""
        return self.concept_storage.get_stored_concepts(session_id)
    
    def get_stored_hypotheses(self, session_id=None, status=None):
        """Get stored hypotheses"""
        return self.hypothesis_storage.get_stored_hypotheses(session_id, status)
    
    def get_stored_models(self, session_id=None):
        """Get stored causal models"""
        return self.causal_storage.get_stored_models(session_id)
    
    def get_stats(self):
        """Get comprehensive storage statistics"""
        concept_stats = self.concept_storage.get_stats()
        hypothesis_stats = self.hypothesis_storage.get_stats()
        model_stats = self.causal_storage.get_stats()
        
        return {
            'concepts_stored': concept_stats['concepts_stored'],
            'hypotheses_stored': hypothesis_stats['hypotheses_stored'],
            'models_stored': model_stats['models_stored'],
            'total_items': (concept_stats['concepts_stored'] + 
                           hypothesis_stats['hypotheses_stored'] + 
                           model_stats['models_stored'])
        }
