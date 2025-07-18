"""
Learning Progress Module
Tracks and manages learning progress metrics for the AGI agent
"""

import time
from typing import Dict, Any


class LearningProgress:
    """Tracks AGI learning progress and metrics"""
    
    def __init__(self, analytics_logger=None):
        # Core learning metrics
        self.progress = {
            'concepts_learned': 0,
            'hypotheses_formed': 0,
            'hypotheses_confirmed': 0,
            'causal_relationships_discovered': 0,
            'patterns_recognized': 0,
            'physics_concepts': 0
        }
        
        # Learning metadata
        self.session_start_time = time.time()
        self.last_update_time = time.time()
        self.learning_events = []
        
        # Analytics logger for persistent statistics
        self.analytics_logger = analytics_logger
    
    def update_concepts(self, count: int):
        """Update concepts learned count"""
        self.progress['concepts_learned'] += count
        self._record_event('concepts_learned', count)
        
        # Update persistent statistics immediately for concepts
        if self.analytics_logger and hasattr(self.analytics_logger, 'update_learning_stats'):
            self.analytics_logger.update_learning_stats(concepts_learned=count)
    
    def update_hypotheses_formed(self, count: int):
        """Update hypotheses formed count"""
        self.progress['hypotheses_formed'] += count
        self._record_event('hypotheses_formed', count)
        
        # Update persistent statistics immediately for hypotheses formed
        if self.analytics_logger and hasattr(self.analytics_logger, 'update_learning_stats'):
            self.analytics_logger.update_learning_stats(hypotheses_formed=count)
    
    def update_hypotheses_confirmed(self, count: int):
        """Update hypotheses confirmed count"""
        self.progress['hypotheses_confirmed'] += count
        self._record_event('hypotheses_confirmed', count)
        
        # Update persistent statistics immediately for confirmed hypotheses (significant milestone)
        if self.analytics_logger and hasattr(self.analytics_logger, 'update_learning_stats'):
            self.analytics_logger.update_learning_stats(hypotheses_confirmed=count)
    
    def update_causal_relationships(self, count: int):
        """Update causal relationships discovered count"""
        self.progress['causal_relationships_discovered'] += count
        self._record_event('causal_relationships_discovered', count)
        
        # Update persistent statistics immediately for causal relationships (significant milestone)
        if self.analytics_logger and hasattr(self.analytics_logger, 'update_learning_stats'):
            self.analytics_logger.update_learning_stats(causal_relationships=count)
    
    def update_patterns(self, count: int):
        """Update patterns recognized count"""
        self.progress['patterns_recognized'] += count
        self._record_event('patterns_recognized', count)
    
    def process_gpu_discoveries(self, gpu_results: Dict[str, Any], cycle_count: int):
        """Process GPU-generated discoveries and update progress"""
        if not gpu_results or 'processed_entities' not in gpu_results:
            return
        
        processed_count = gpu_results.get('processed_entities', 0)
        if processed_count <= 0:
            return
        
        # Update concepts with GPU discoveries
        self.update_concepts(processed_count)
        
        # Generate hypotheses based on significant patterns
        if cycle_count % 10 == 0 and processed_count > 50:
            hypotheses_count = min(processed_count // 100, 5)
            if hypotheses_count > 0:
                self.update_hypotheses_formed(hypotheses_count)
        
        # Confirm hypotheses periodically
        if cycle_count % 20 == 0 and self.progress['hypotheses_formed'] > 0:
            confirmed_count = min(self.progress['hypotheses_formed'] // 2, 3)
            if confirmed_count > 0:
                self.update_hypotheses_confirmed(confirmed_count)
                self.update_causal_relationships(confirmed_count)
    
    def _record_event(self, event_type: str, value: int):
        """Record a learning event"""
        event = {
            'type': event_type,
            'value': value,
            'timestamp': time.time(),
            'session_time': time.time() - self.session_start_time
        }
        self.learning_events.append(event)
        self.last_update_time = time.time()
        
        # Keep events manageable
        if len(self.learning_events) > 1000:
            self.learning_events = self.learning_events[-500:]
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current learning progress"""
        return self.progress.copy()
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update learning metrics from external source"""
        if not isinstance(metrics, dict):
            return
        
        for key, value in metrics.items():
            if key in self.progress and isinstance(value, (int, float)):
                # For most metrics, we want to set the absolute value from external modules
                # But preserve accumulated values for core learning metrics
                if key in ['concepts_learned', 'hypotheses_formed', 'hypotheses_confirmed', 
                          'causal_relationships_discovered', 'patterns_recognized', 'physics_concepts']:
                    # For core learning metrics, take the maximum to avoid overwriting accumulated progress
                    self.progress[key] = max(self.progress[key], int(value))
                else:
                    # For other metrics, set absolute value
                    self.progress[key] = int(value)
                self._record_event(key, int(value))
        
        self.last_update_time = time.time()
    
    def get_learning_rate(self) -> float:
        """Calculate current learning rate"""
        session_time = time.time() - self.session_start_time
        total_learning = sum(self.progress.values())
        return total_learning / max(session_time, 1.0)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary"""
        session_time = time.time() - self.session_start_time
        return {
            'progress': self.progress.copy(),
            'session_duration': session_time,
            'learning_rate': self.get_learning_rate(),
            'total_events': len(self.learning_events),
            'last_update': self.last_update_time
        }
    
    def reset_progress(self):
        """Reset all learning progress"""
        self.progress = {
            'concepts_learned': 0,
            'hypotheses_formed': 0,
            'hypotheses_confirmed': 0,
            'causal_relationships_discovered': 0,
            'patterns_recognized': 0,
            'physics_concepts': 0
        }
        self.session_start_time = time.time()
        self.last_update_time = time.time()
        self.learning_events.clear()
    
    def save_session_to_persistent_stats(self):
        """Save current session progress to persistent statistics (called on session end)"""
        if self.analytics_logger and hasattr(self.analytics_logger, 'update_learning_stats'):
            # Update persistent stats with current session totals
            self.analytics_logger.update_learning_stats(
                concepts_learned=self.progress['concepts_learned'],
                hypotheses_formed=self.progress['hypotheses_formed'], 
                hypotheses_confirmed=self.progress['hypotheses_confirmed'],
                causal_relationships=self.progress['causal_relationships_discovered']
            )
            print(f"💾 [Session] Saved session progress to persistent stats: {self.progress['concepts_learned']} concepts")
