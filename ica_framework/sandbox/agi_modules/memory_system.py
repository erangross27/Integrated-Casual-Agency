"""
Memory System Module
Handles short-term, long-term, and episodic memory for the AGI agent
"""

import time
from typing import Dict, List, Any
from collections import deque


class MemorySystem:
    """Manages different types of memory for the AGI agent"""
    
    def __init__(self, short_term_size: int = 100):
        # Memory systems
        self.short_term_memory = deque(maxlen=short_term_size)
        self.long_term_memory = []
        self.episodic_memory = []
        
        # Memory metadata
        self.memory_stats = {
            'total_memories': 0,
            'consolidations': 0,
            'retrievals': 0
        }
    
    def store_short_term(self, memory: Dict[str, Any]):
        """Store memory in short-term memory with automatic consolidation"""
        memory['timestamp'] = time.time()
        memory['memory_type'] = 'short_term'
        self.short_term_memory.append(memory)
        self.memory_stats['total_memories'] += 1
        
        # Auto-consolidate if short-term memory is getting full (80% capacity)
        if len(self.short_term_memory) >= int(self.short_term_memory.maxlen * 0.8):
            self._auto_consolidate_memories()
    
    def _auto_consolidate_memories(self):
        """Automatically consolidate important memories to long-term storage"""
        try:
            # Consolidate older memories with high confidence or importance
            for memory in list(self.short_term_memory)[:10]:  # Check oldest 10 memories
                confidence = memory.get('confidence', 0)
                importance = memory.get('importance', 0)
                
                # Consolidate high-confidence or important memories
                if confidence > 0.7 or importance > 0.8:
                    self.consolidate_memory(memory)
                    
        except Exception as e:
            # Ignore consolidation errors to prevent disruption
            pass
    
    def store_long_term(self, memory: Dict[str, Any]):
        """Store memory in long-term memory"""
        memory['timestamp'] = time.time()
        memory['memory_type'] = 'long_term'
        self.long_term_memory.append(memory)
        self.memory_stats['total_memories'] += 1
    
    def store_episodic(self, episode: Dict[str, Any]):
        """Store episodic memory"""
        episode['timestamp'] = time.time()
        episode['memory_type'] = 'episodic'
        self.episodic_memory.append(episode)
        self.memory_stats['total_memories'] += 1
    
    def get_recent_memories(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent memories from short-term memory"""
        recent = list(self.short_term_memory)[-count:]
        self.memory_stats['retrievals'] += len(recent)
        return recent
    
    def consolidate_memory(self, memory: Dict[str, Any]):
        """Move important memory from short-term to long-term"""
        try:
            # Always consolidate to long-term (simplified approach)
            long_term_copy = memory.copy()
            long_term_copy['memory_type'] = 'long_term'
            long_term_copy['consolidated_at'] = time.time()
            
            # Store in long-term
            self.store_long_term(long_term_copy)
            self.memory_stats['consolidations'] += 1
            
            # Don't try to remove from short-term deque (too complex and error-prone)
            # Let it naturally age out when deque fills up
            
        except Exception as e:
            # Fallback: simple consolidation
            try:
                simple_copy = {
                    'type': memory.get('type', 'unknown'),
                    'consolidated_from': 'short_term',
                    'consolidated_at': time.time(),
                    'original_data': memory
                }
                self.store_long_term(simple_copy)
                self.memory_stats['consolidations'] += 1
            except:
                pass  # Silent fail to prevent disruption
    
    def search_memories(self, query_type: str) -> List[Dict[str, Any]]:
        """Search memories by type or content"""
        results = []
        
        # Search short-term memory
        for memory in self.short_term_memory:
            if query_type in memory.get('type', ''):
                results.append(memory)
        
        # Search long-term memory
        for memory in self.long_term_memory:
            if query_type in memory.get('type', ''):
                results.append(memory)
        
        self.memory_stats['retrievals'] += len(results)
        return results
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics"""
        return {
            'short_term': len(self.short_term_memory),
            'long_term': len(self.long_term_memory),
            'episodic': len(self.episodic_memory),
            'total_stored': self.memory_stats['total_memories'],
            'consolidations': self.memory_stats['consolidations'],
            'retrievals': self.memory_stats['retrievals']
        }
    
    def recall_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        """Recall memories by type"""
        return self.search_memories(memory_type)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary"""
        usage = self.get_memory_usage()
        return {
            'usage': usage,
            'short_term': len(self.short_term_memory),  # Direct access for monitoring
            'long_term': len(self.long_term_memory),    # Direct access for monitoring  
            'episodic': len(self.episodic_memory),      # Direct access for monitoring
            'consolidations': self.memory_stats['consolidations'],  # Direct access for monitoring
            'total_memories': self.memory_stats['total_memories'],  # Use actual total count
            'recent_memories_count': len(self.get_recent_memories()),
            'memory_stats': self.memory_stats.copy(),
            'working_memory_usage': (len(self.short_term_memory) / self.short_term_memory.maxlen) * 100,
            'episodic_count': len(self.episodic_memory)
        }
    
    def store_experience(self, experience: Dict[str, Any]):
        """Store learning experience in appropriate memory system"""
        if not isinstance(experience, dict):
            return
        
        # Determine memory type based on experience content
        if 'learning_context' in experience or 'discovery_hints' in experience:
            # This is a significant learning experience - store as episodic
            self.store_episodic(experience)
        else:
            # Regular experience - store in short-term
            self.store_short_term(experience)
    
    def clear_memories(self):
        """Clear all memories"""
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.episodic_memory.clear()
        self.memory_stats = {
            'total_memories': 0,
            'consolidations': 0,
            'retrievals': 0
        }
