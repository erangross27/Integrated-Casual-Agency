#!/usr/bin/env python3
"""
Session Management Module
Handles session state and metadata persistence
"""

import json
import time


class SessionManager:
    """Handles session state and metadata persistence"""
    
    def __init__(self, knowledge_graph, session_id):
        self.kg = knowledge_graph
        self.session_id = session_id
        self.sessions_stored = 0
        
        # Create session start record
        self._create_session_start()
    
    def _create_session_start(self):
        """Create session start record"""
        try:
            session_start_entity = {
                'id': f"session_start_{self.session_id}",
                'type': 'session_start',
                'session_id': self.session_id,
                'start_timestamp': time.time(),
                'status': 'active'
            }
            
            if self.kg.add_entity(session_start_entity):
                print(f"💾 [SESSION] Session {self.session_id} started")
                
        except Exception as e:
            print(f"[SESSION] ⚠️ Session start error: {e}")
    
    def save_session_state(self, agi_agent, gpu_stats=None):
        """Save complete session state for restoration"""
        if not agi_agent or not self.kg:
            return False
        
        try:
            # Get comprehensive session state
            session_state = {
                'session_id': self.session_id,
                'timestamp': time.time(),
                'learning_progress': getattr(agi_agent, 'learning_progress', {}),
                'memory_state': {
                    'short_term': list(getattr(agi_agent, 'short_term_memory', [])),
                    'long_term': getattr(agi_agent, 'long_term_memory', [])
                },
                'gpu_stats': gpu_stats or {},
                'simulation_steps': getattr(agi_agent, 'simulation_steps', 0),
                'concepts_learned': getattr(agi_agent, 'learning_progress', {}).get('concepts_learned', 0),
                'hypotheses_formed': getattr(agi_agent, 'learning_progress', {}).get('hypotheses_formed', 0),
                'causal_relationships_discovered': getattr(agi_agent, 'learning_progress', {}).get('causal_relationships_discovered', 0)
            }
            
            session_entity = {
                'type': 'session_state',
                'session_id': self.session_id,
                'state_data': json.dumps(session_state),
                'timestamp': time.time(),
                'is_active': True
            }
            
            # Mark previous sessions as inactive
            if hasattr(self.kg, 'execute_custom_query'):
                try:
                    self.kg.execute_custom_query(
                        f"MATCH (n:Entity {{type: 'session_state', session_id: '{self.session_id}'}}) SET n.is_active = false"
                    )
                except:
                    pass
            
            session_id = f"session_state_{self.session_id}_{int(time.time())}"
            if self.kg.add_entity(session_id, 'session_state', session_entity):
                self.sessions_stored += 1
                return True
            
        except Exception as e:
            print(f"[SESSION] ⚠️ Session state save error: {e}")
            return False
    
    def restore_session(self, session_id=None):
        """Restore session from database"""
        if not self.kg:
            return None
        
        try:
            # Find latest session if no ID provided
            if not session_id:
                query = "MATCH (n:Entity {type: 'session_state'}) RETURN n ORDER BY n.timestamp DESC LIMIT 1"
            else:
                query = f"MATCH (n:Entity {{type: 'session_state', session_id: '{session_id}'}}) RETURN n ORDER BY n.timestamp DESC LIMIT 1"
            
            result = self.kg.execute_custom_query(query)
            if result and len(result) > 0:
                session_data = result[0].get('n', {})
                if session_data:
                    state_data = json.loads(session_data.get('state_data', '{}'))
                    print(f"💾 [SESSION] Restored session metadata")
                    print(f"💾 [SESSION] Concepts: {state_data.get('concepts_learned', 0)}")
                    print(f"💾 [SESSION] Hypotheses: {state_data.get('hypotheses_formed', 0)}")
                    print(f"💾 [SESSION] Simulation steps: {state_data.get('simulation_steps', 0)}")
                    return state_data
            
        except Exception as e:
            print(f"[SESSION] ⚠️ Session restore error: {e}")
        
        return None
    
    def mark_session_end(self, agi_agent):
        """Mark session as ended"""
        if not agi_agent or not self.kg:
            return False
        
        try:
            session_end_entity = {
                'id': f"session_end_{self.session_id}_{int(time.time())}",
                'type': 'session_end',
                'session_id': self.session_id,
                'end_timestamp': time.time(),
                'final_learning_progress': json.dumps(getattr(agi_agent, 'learning_progress', {})),
                'final_simulation_steps': getattr(agi_agent, 'simulation_steps', 0)
            }
            
            # Mark session start as inactive
            if hasattr(self.kg, 'execute_custom_query'):
                try:
                    self.kg.execute_custom_query(
                        f"MATCH (n:Entity {{type: 'session_start', session_id: '{self.session_id}'}}) SET n.status = 'ended'"
                    )
                except:
                    pass
            
            if self.kg.add_entity(session_end_entity):
                print(f"💾 [SESSION] Session {self.session_id} ended")
                return True
            
        except Exception as e:
            print(f"[SESSION] ⚠️ Session end error: {e}")
            return False
    
    def get_session_history(self, limit=10):
        """Get session history"""
        if not self.kg:
            return []
        
        try:
            query = f"MATCH (n:Entity {{type: 'session_start'}}) RETURN n ORDER BY n.start_timestamp DESC LIMIT {limit}"
            result = self.kg.execute_custom_query(query)
            
            sessions = []
            for record in result:
                session_data = record.get('n', {})
                sessions.append({
                    'session_id': session_data.get('session_id', 'unknown'),
                    'start_timestamp': session_data.get('start_timestamp', 0),
                    'status': session_data.get('status', 'unknown')
                })
            
            return sessions
            
        except Exception as e:
            print(f"[SESSION] ⚠️ Session history error: {e}")
            return []
    
    def get_stats(self):
        """Get session management statistics"""
        return {
            'sessions_stored': self.sessions_stored,
            'current_session_id': self.session_id
        }
