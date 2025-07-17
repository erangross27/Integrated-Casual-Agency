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
        """Store memory in short-term memory"""
        memory['timestamp'] = time.time()
        memory['memory_type'] = 'short_term'
        self.short_term_memory.append(memory)
        self.memory_stats['total_memories'] += 1
    
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
        if memory in self.short_term_memory:
            self.store_long_term(memory.copy())
            self.memory_stats['consolidations'] += 1
    
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
    
    def store_experience(self, experience: Dict[str, Any]):
        """Store an experience in appropriate memory system"""
        # Add to short-term memory first
        self.store_memory('short_term', experience)
        
        # If high importance, also add to episodic memory
        importance = experience.get('importance', 0.5)
        if importance > 0.7:
            self.store_memory('episodic', experience)
    
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
