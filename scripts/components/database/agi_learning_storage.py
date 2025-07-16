#!/usr/bin/env python3
"""
AGI Learning Storage Module
Simplified coordinator that uses modular components
"""

from .agi_storage import AGILearningCoordinator


class AGILearningStorage:
    """Simplified AGI Learning Storage using modular components"""
    
    def __init__(self, knowledge_graph, session_id):
        self.coordinator = AGILearningCoordinator(knowledge_graph, session_id)
        self.kg = knowledge_graph
        self.session_id = session_id
    
    def store_agi_learning(self, agi_agent):
        """Store AGI learning data using modular coordinator"""
        return self.coordinator.store_agi_learning(agi_agent)
    
    def get_stored_concepts(self, session_id=None):
        """Get stored concepts"""
        return self.coordinator.get_stored_concepts(session_id)
    
    def get_stored_hypotheses(self, session_id=None):
        """Get stored hypotheses"""
        return self.coordinator.get_stored_hypotheses(session_id)
    
    def restore_agi_concepts(self, agi_agent):
        """Restore AGI concepts from database to agent"""
        return self.coordinator.restore_agi_learning(agi_agent)
    
    def get_stats(self):
        """Get AGI learning storage statistics"""
        return self.coordinator.get_stats()
    
    # Compatibility properties
    @property
    def concepts_stored(self):
        return self.coordinator.get_stats()['concepts_stored']
    
    @property
    def hypotheses_stored(self):
        return self.coordinator.get_stats()['hypotheses_stored']
    
    @property
    def relationships_stored(self):
        return self.coordinator.get_stats()['models_stored']
