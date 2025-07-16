#!/usr/bin/env python3
"""
Learning State Persistence Module
Handles saving and restoring complete AGI learning states
"""

import json
import time
from collections import deque


class LearningStatePersistence:
    """Handles complete AGI learning state persistence"""
    
    def __init__(self, knowledge_graph, session_id):
        self.kg = knowledge_graph
        self.session_id = session_id
        self.states_stored = 0
    
    def save_learning_state(self, agi_agent):
        """Save complete AGI learning state"""
        if not agi_agent or not self.kg:
            return False
        
        try:
            # Capture ALL learning state
            learning_state = {
                'knowledge_base': getattr(agi_agent, 'knowledge_base', {}),
                'causal_models': getattr(agi_agent, 'causal_models', {}),
                'active_hypotheses': getattr(agi_agent, 'active_hypotheses', []),
                'tested_hypotheses': getattr(agi_agent, 'tested_hypotheses', []),
                'learning_progress': getattr(agi_agent, 'learning_progress', {}),
                'short_term_memory': list(getattr(agi_agent, 'short_term_memory', [])),
                'long_term_memory': getattr(agi_agent, 'long_term_memory', []),
                'curiosity_level': getattr(agi_agent, 'curiosity_level', 0.5),
                'exploration_rate': getattr(agi_agent, 'exploration_rate', 0.3),
                'novelty_threshold': getattr(agi_agent, 'novelty_threshold', 0.7),
                'simulation_steps': getattr(agi_agent, 'simulation_steps', 0),
                'concepts_learned': getattr(agi_agent, 'learning_progress', {}).get('concepts_learned', 0),
                'hypotheses_formed': getattr(agi_agent, 'learning_progress', {}).get('hypotheses_formed', 0),
                'causal_relationships_discovered': getattr(agi_agent, 'learning_progress', {}).get('causal_relationships_discovered', 0)
            }
            
            learning_entity = {
                'type': 'learning_state',
                'session_id': self.session_id,
                'learning_data': json.dumps(learning_state),
                'timestamp': time.time(),
                'is_current': True
            }
            
            # Mark previous states as not current
            if hasattr(self.kg, 'execute_custom_query'):
                try:
                    self.kg.execute_custom_query(
                        f"MATCH (n:Entity {{type: 'learning_state'}}) SET n.is_current = false"
                    )
                except:
                    pass
            
            learning_id = f"learning_state_{self.session_id}_{int(time.time())}"
            if self.kg.add_entity(learning_id, 'learning_state', learning_entity):
                self.states_stored += 1
                print(f"💾 [LEARNING] Saved complete learning state ({len(json.dumps(learning_state))} bytes)")
                print(f"💾 [LEARNING] Knowledge: {len(learning_state['knowledge_base'])} items")
                print(f"💾 [LEARNING] Hypotheses: {len(learning_state['active_hypotheses'])} active")
                print(f"💾 [LEARNING] Progress: {learning_state['concepts_learned']} concepts, {learning_state['hypotheses_formed']} hypotheses")
                return True
            
        except Exception as e:
            print(f"[LEARNING] ⚠️ Learning state save error: {e}")
            return False
    
    def restore_learning_state(self, agi_agent):
        """Restore complete AGI learning state"""
        if not agi_agent or not self.kg:
            return False
        
        try:
            # Get latest learning state
            query = f"MATCH (n:Entity {{type: 'learning_state', is_current: true}}) RETURN n ORDER BY n.timestamp DESC LIMIT 1"
            result = self.kg.execute_custom_query(query)
            
            if result and len(result) > 0:
                learning_data = result[0].get('n', {})
                learning_state = json.loads(learning_data.get('learning_data', '{}'))
                
                # Restore ALL AGI agent state
                if hasattr(agi_agent, 'knowledge_base'):
                    agi_agent.knowledge_base = learning_state.get('knowledge_base', {})
                if hasattr(agi_agent, 'causal_models'):
                    agi_agent.causal_models = learning_state.get('causal_models', {})
                if hasattr(agi_agent, 'active_hypotheses'):
                    agi_agent.active_hypotheses = learning_state.get('active_hypotheses', [])
                if hasattr(agi_agent, 'tested_hypotheses'):
                    agi_agent.tested_hypotheses = learning_state.get('tested_hypotheses', [])
                if hasattr(agi_agent, 'learning_progress'):
                    agi_agent.learning_progress.update(learning_state.get('learning_progress', {}))
                if hasattr(agi_agent, 'short_term_memory'):
                    agi_agent.short_term_memory = deque(learning_state.get('short_term_memory', []), maxlen=100)
                if hasattr(agi_agent, 'long_term_memory'):
                    agi_agent.long_term_memory = learning_state.get('long_term_memory', [])
                if hasattr(agi_agent, 'curiosity_level'):
                    agi_agent.curiosity_level = learning_state.get('curiosity_level', 0.5)
                if hasattr(agi_agent, 'exploration_rate'):
                    agi_agent.exploration_rate = learning_state.get('exploration_rate', 0.3)
                if hasattr(agi_agent, 'novelty_threshold'):
                    agi_agent.novelty_threshold = learning_state.get('novelty_threshold', 0.7)
                if hasattr(agi_agent, 'simulation_steps'):
                    agi_agent.simulation_steps = learning_state.get('simulation_steps', 0)
                
                print(f"💾 [LEARNING] Restored complete learning state")
                print(f"💾 [LEARNING] Knowledge base: {len(agi_agent.knowledge_base)} items")
                print(f"💾 [LEARNING] Causal models: {len(agi_agent.causal_models)} models")
                print(f"💾 [LEARNING] Active hypotheses: {len(agi_agent.active_hypotheses)} hypotheses")
                print(f"💾 [LEARNING] Learning progress: {agi_agent.learning_progress}")
                print(f"💾 [LEARNING] Simulation steps: {getattr(agi_agent, 'simulation_steps', 0)}")
                
                return True
            else:
                print(f"💾 [LEARNING] No previous learning state found")
                return False
            
        except Exception as e:
            print(f"[LEARNING] ⚠️ Learning state restore error: {e}")
            return False
    
    def get_latest_learning_summary(self):
        """Get summary of latest learning state"""
        if not self.kg:
            return None
        
        try:
            query = f"MATCH (n:Entity {{type: 'learning_state', is_current: true}}) RETURN n ORDER BY n.timestamp DESC LIMIT 1"
            result = self.kg.execute_custom_query(query)
            
            if result and len(result) > 0:
                learning_data = result[0].get('n', {})
                learning_state = json.loads(learning_data.get('learning_data', '{}'))
                
                return {
                    'concepts_learned': learning_state.get('concepts_learned', 0),
                    'hypotheses_formed': learning_state.get('hypotheses_formed', 0),
                    'causal_relationships_discovered': learning_state.get('causal_relationships_discovered', 0),
                    'simulation_steps': learning_state.get('simulation_steps', 0),
                    'knowledge_base_size': len(learning_state.get('knowledge_base', {})),
                    'active_hypotheses_count': len(learning_state.get('active_hypotheses', [])),
                    'timestamp': learning_data.get('timestamp', 0)
                }
            
        except Exception as e:
            print(f"[LEARNING] ⚠️ Learning summary error: {e}")
        
        return None
    
    def get_stats(self):
        """Get learning state persistence statistics"""
        return {
            'states_stored': self.states_stored
        }
