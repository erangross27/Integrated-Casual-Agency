#!/usr/bin/env python3
"""
Hypothesis Storage Module
Handles storage and retrieval of AGI hypotheses
"""

import time


class HypothesisStorage:
    """Handles AGI hypothesis storage operations"""
    
    def __init__(self, knowledge_graph, session_id):
        self.kg = knowledge_graph
        self.session_id = session_id
        self.hypotheses_stored = 0
    
    def store_active_hypotheses(self, active_hypotheses):
        """Store active hypotheses"""
        if not active_hypotheses or len(active_hypotheses) == 0:
            return 0
        
        stored_count = 0
        
        for i, hypothesis in enumerate(active_hypotheses):
            hypothesis_id = f"agi_hypothesis_{i}_{int(time.time() * 1000)}_{self.session_id}"
            hypothesis_entity = {
                'type': 'agi_hypothesis',
                'description': str(hypothesis)[:100],
                'status': 'active',
                'created_at': time.time(),
                'session_id': self.session_id,
                'source': 'agi_learning'
            }
            
            if self.kg.add_entity(hypothesis_id, 'agi_hypothesis', hypothesis_entity):
                stored_count += 1
        
        self.hypotheses_stored += stored_count
        return stored_count
    
    def store_confirmed_hypotheses(self, confirmed_hypotheses):
        """Store confirmed hypotheses"""
        if not confirmed_hypotheses or len(confirmed_hypotheses) == 0:
            return 0
        
        stored_count = 0
        
        for i, hypothesis in enumerate(confirmed_hypotheses):
            hypothesis_id = f"agi_confirmed_hypothesis_{i}_{int(time.time() * 1000)}_{self.session_id}"
            hypothesis_entity = {
                'type': 'agi_hypothesis',
                'description': str(hypothesis)[:100],
                'status': 'confirmed',
                'created_at': time.time(),
                'session_id': self.session_id,
                'source': 'agi_learning'
            }
            
            if self.kg.add_entity(hypothesis_id, 'agi_hypothesis', hypothesis_entity):
                stored_count += 1
        
        self.hypotheses_stored += stored_count
        return stored_count
    
    def get_stored_hypotheses(self, session_id=None, status=None):
        """Get stored hypotheses from database"""
        if not self.kg:
            return []
        
        try:
            # Build query based on filters
            conditions = ["type: 'agi_hypothesis'"]
            
            if session_id:
                conditions.append(f"session_id: '{session_id}'")
            
            if status:
                conditions.append(f"status: '{status}'")
            
            query = f"MATCH (n:Entity {{{', '.join(conditions)}}}) RETURN n ORDER BY n.created_at DESC"
            
            result = self.kg.execute_custom_query(query)
            
            hypotheses = []
            for record in result:
                hypothesis_data = record.get('n', {})
                hypotheses.append({
                    'id': hypothesis_data.get('id', 'unknown'),
                    'description': hypothesis_data.get('description', 'No description'),
                    'status': hypothesis_data.get('status', 'unknown'),
                    'created_at': hypothesis_data.get('created_at', 0),
                    'session_id': hypothesis_data.get('session_id', 'unknown')
                })
            
            return hypotheses
            
        except Exception as e:
            print(f"[HYPOTHESIS] ⚠️ Hypothesis retrieval error: {e}")
            return []
    
    def restore_hypotheses_to_agent(self, agi_agent):
        """Restore hypotheses from database to AGI agent"""
        if not self.kg or not agi_agent:
            return False
        
        try:
            # Get all stored hypotheses from any session (for restoration)
            active_hypotheses = self.get_stored_hypotheses(None, 'active')  # No session_id = get all
            confirmed_hypotheses = self.get_stored_hypotheses(None, 'confirmed')  # No session_id = get all
            
            # Restore to agent if it has hypothesis storage
            if hasattr(agi_agent, 'active_hypotheses'):
                if not hasattr(agi_agent.active_hypotheses, 'clear'):
                    agi_agent.active_hypotheses = []
                
                for hyp in active_hypotheses:
                    if hyp['description'] not in agi_agent.active_hypotheses:
                        agi_agent.active_hypotheses.append(hyp['description'])
            
            if hasattr(agi_agent, 'confirmed_hypotheses'):
                if not hasattr(agi_agent.confirmed_hypotheses, 'clear'):
                    agi_agent.confirmed_hypotheses = []
                
                for hyp in confirmed_hypotheses:
                    if hyp['description'] not in agi_agent.confirmed_hypotheses:
                        agi_agent.confirmed_hypotheses.append(hyp['description'])
            
            if len(active_hypotheses) > 0 or len(confirmed_hypotheses) > 0:
                print(f"💾 [HYPOTHESIS] ✅ Restored {len(active_hypotheses)} active + {len(confirmed_hypotheses)} confirmed hypotheses")
                return True
            
            return False
            
        except Exception as e:
            print(f"[HYPOTHESIS] ⚠️ Hypothesis restoration error: {e}")
            return False
    
    def get_stats(self):
        """Get hypothesis storage statistics"""
        return {
            'hypotheses_stored': self.hypotheses_stored
        }
