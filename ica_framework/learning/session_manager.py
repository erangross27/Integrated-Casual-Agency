"""
Session Manager for ICA Framework Learning
Handles session persistence, progress tracking, and checkpoint management
"""

import json
import time
import signal
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from ..enhanced_knowledge_graph import EnhancedKnowledgeGraph
    HAS_ENHANCED_KG = True
except ImportError:
    HAS_ENHANCED_KG = False


class SessionManager:
    """Manages learning session state, checkpoints, and progress tracking"""
    
    def __init__(self, database_backend: str = "memory", database_config: Dict[str, Any] = None):
        self.database_backend = database_backend
        self.database_config = database_config or {}
        self.running = True
        
        # Session tracking
        self.session_start_time = datetime.now()
        self.session_timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        
        self.session_stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'experiments_conducted': 0,
            'confidence_progression': [],
            'learning_events': [],
            'scenarios_completed': 0,
            'session_start_time': time.time(),
            'total_learning_time': 0.0,
            'session_id': self.session_timestamp,
            'database_backend': database_backend,
            'workers_used': 0,
            'total_batches_processed': 0
        }
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def set_agent_reference(self, agent):
        """Store agent reference for checkpoint operations"""
        self.agent = agent
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n🛑 Stopping... Neo4j has all data")
        self.running = False
        # Use stored agent reference if available
        agent = getattr(self, 'agent', None)
        self.save_checkpoint(agent)
        print("💾 All knowledge preserved in Neo4j database")
        os._exit(0)
    
    def load_checkpoint(self, agent) -> Optional[Dict[str, Any]]:
        """Load previous learning session from Neo4j"""
        if self.database_backend == "neo4j" and HAS_ENHANCED_KG:
            try:
                if (hasattr(agent.knowledge_graph, 'db') and 
                    hasattr(agent.knowledge_graph.db, 'execute_query')):
                    
                    session_query = """
                    MATCH (session:SessionMeta)
                    RETURN session.scenarios_completed as scenarios_completed,
                           session.total_learning_time as total_learning_time,
                           session.session_id as session_id,
                           session.last_updated as last_updated,
                           session.total_nodes as total_nodes,
                           session.total_edges as total_edges
                    ORDER BY session.last_updated DESC
                    LIMIT 1
                    """
                    
                    result = agent.knowledge_graph.db.execute_query(session_query, {})
                    if result and result[0]:
                        session_data = result[0]
                        print(f"📈 Found saved metadata: {session_data['scenarios_completed']} scenarios completed")
                        print(f"⏱️ Restored learning time: {session_data['total_learning_time']:.1f}s")
                        print(f"📊 Restored nodes: {session_data.get('total_nodes', 0)} | edges: {session_data.get('total_edges', 0)}")
                        print(f"🆔 Continuing session: {session_data['session_id']}")
                        
                        # Update session stats with loaded values (with null handling)
                        self.session_stats['total_nodes'] = session_data.get('total_nodes') or 0
                        self.session_stats['total_edges'] = session_data.get('total_edges') or 0
                        self.session_stats['scenarios_completed'] = session_data.get('scenarios_completed') or 0
                        self.session_stats['total_learning_time'] = session_data.get('total_learning_time') or 0.0
                        
                        # Preserve original session_id if continuing same session
                        if session_data.get('session_id'):
                            self.session_stats['session_id'] = session_data['session_id']
                            
                        return session_data
            except Exception as e:
                print(f"⚠️ Could not load session metadata: {e}")
        
        print("📂 Starting fresh session")
        return None
    
    def save_checkpoint(self, agent=None):
        """Save current learning progress to Neo4j"""
        try:
            if (self.database_backend == "neo4j" and HAS_ENHANCED_KG and agent and
                hasattr(agent.knowledge_graph, 'db') and 
                hasattr(agent.knowledge_graph.db, 'execute_query')):
                
                # Test connection first
                try:
                    test_query = "RETURN 1 AS test"
                    agent.knowledge_graph.db.execute_query(test_query, {})
                except Exception as e:
                    print(f"⚠️ Database connection test failed: {e}")
                    return False
                
                # Save session metadata
                session_query = """
                MERGE (session:SessionMeta {session_id: $session_id})
                SET session.scenarios_completed = $scenarios_completed,
                    session.total_nodes = $total_nodes,
                    session.total_edges = $total_edges,
                    session.total_learning_time = $total_learning_time,
                    session.last_updated = $last_updated,
                    session.database_backend = $database_backend
                RETURN session
                """
                
                parameters = {
                    'session_id': self.session_stats['session_id'],
                    'scenarios_completed': self.session_stats['scenarios_completed'],
                    'total_nodes': self.session_stats['total_nodes'],
                    'total_edges': self.session_stats['total_edges'],
                    'total_learning_time': self.session_stats['total_learning_time'],
                    'last_updated': time.time(),
                    'database_backend': self.database_backend
                }
                
                agent.knowledge_graph.db.execute_query(session_query, parameters)
                time.sleep(0.05)  # Brief pause for transaction completion
                
                print(f"📊 Session metadata saved to Neo4j: {self.session_stats['scenarios_completed']} scenarios | {self.session_stats['total_nodes']} nodes | {self.session_stats['total_edges']} edges")
                return True
            
            else:
                print(f"📊 Session stats: {self.session_stats['scenarios_completed']} scenarios")
                return True
        
        except Exception as e:
            print(f"⚠️ Session metadata save error: {e}")
            return False
    
    def update_stats(self, **kwargs):
        """Update session statistics"""
        for key, value in kwargs.items():
            if key in self.session_stats:
                # Ensure numeric values are properly typed
                if key in ['total_nodes', 'total_edges', 'scenarios_completed', 'workers_used', 'total_batches_processed']:
                    self.session_stats[key] = int(value) if value is not None else 0
                elif key in ['total_learning_time']:
                    self.session_stats[key] = float(value) if value is not None else 0.0
                else:
                    self.session_stats[key] = value
    
    def add_learning_event(self, event: Dict[str, Any]):
        """Add a learning event to the session log"""
        self.session_stats['learning_events'].append(event)
        
        # Keep only last 1000 events to prevent memory bloat
        if len(self.session_stats['learning_events']) > 1000:
            self.session_stats['learning_events'] = self.session_stats['learning_events'][-1000:]
    
    def should_show_detailed_update(self, edges_count: int, last_detailed_update: int) -> bool:
        """Determine if we should show a detailed update based on edge milestones"""
        edge_milestone = (edges_count // 2000) > (last_detailed_update // 2000)
        return edge_milestone
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session"""
        runtime = time.time() - self.session_stats['session_start_time']
        learning_rate = self.session_stats['scenarios_completed'] / max(self.session_stats['total_learning_time'], 0.001)
        
        return {
            'session_id': self.session_stats['session_id'],
            'runtime': runtime,
            'scenarios_completed': self.session_stats['scenarios_completed'],
            'total_nodes': self.session_stats['total_nodes'],
            'total_edges': self.session_stats['total_edges'],
            'learning_rate': learning_rate,
            'database_backend': self.database_backend
        }
