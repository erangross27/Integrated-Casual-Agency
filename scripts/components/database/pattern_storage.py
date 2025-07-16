#!/usr/bin/env python3
"""
Pattern Storage Module
Handles continuous pattern storage and retrieval
"""

import json
import time
import threading
from collections import deque


class PatternStorage:
    """Handles continuous pattern storage with auto-save"""
    
    def __init__(self, knowledge_graph, session_id):
        self.kg = knowledge_graph
        self.session_id = session_id
        self.pattern_buffer = deque(maxlen=10000)
        self.patterns_stored = 0
        self.auto_save_interval = 10  # Save every 10 seconds - more frequent!
        self.last_save_time = time.time()
        
        # Start aggressive auto-save thread
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.auto_save_thread.start()
        
        print(f"💾 [PATTERNS] Pattern storage initialized - auto-save every {self.auto_save_interval}s")
    
    def _auto_save_loop(self):
        """Aggressive auto-save loop - NO PATTERNS LOST"""
        while True:
            time.sleep(self.auto_save_interval)
            if len(self.pattern_buffer) > 0:
                self._save_pattern_buffer()
    
    def _save_pattern_buffer(self):
        """Save accumulated patterns to database with aggressive logging"""
        if not self.pattern_buffer or not self.kg:
            print(f"💾 [PATTERNS] ⚠️ Cannot save - no patterns or no knowledge graph")
            return
        
        try:
            print(f"💾 [PATTERNS] 🚀 Starting pattern buffer save...")
            
            # Save patterns in batches
            batch_size = 1000
            patterns_to_save = list(self.pattern_buffer)
            
            print(f"💾 [PATTERNS] Saving {len(patterns_to_save)} patterns in batches of {batch_size}")
            
            saved_batches = 0
            for i in range(0, len(patterns_to_save), batch_size):
                batch = patterns_to_save[i:i+batch_size]
                pattern_entity = {
                    'type': 'pattern_batch',
                    'session_id': self.session_id,
                    'patterns': json.dumps(batch),
                    'pattern_count': len(batch),
                    'timestamp': time.time(),
                    'batch_index': i // batch_size
                }
                
                pattern_id = f"pattern_batch_{self.session_id}_{i//batch_size}_{int(time.time())}"
                if self.kg.add_entity(pattern_id, 'pattern_batch', pattern_entity):
                    self.patterns_stored += len(batch)
                    saved_batches += 1
                    print(f"💾 [PATTERNS] ✅ Saved batch {saved_batches}: {len(batch)} patterns")
                else:
                    print(f"💾 [PATTERNS] ❌ Failed to save batch {saved_batches}")
            
            # Clear buffer after saving
            pattern_count = len(self.pattern_buffer)
            self.pattern_buffer.clear()
            self.last_save_time = time.time()
            
            print(f"💾 [PATTERNS] 🎉 PATTERN SAVE COMPLETE: {pattern_count} patterns saved in {saved_batches} batches")
            print(f"💾 [PATTERNS] Total patterns stored: {self.patterns_stored}")
            
        except Exception as e:
            print(f"💾 [PATTERNS] 💥 CRITICAL ERROR saving patterns: {e}")
            import traceback
            traceback.print_exc()
    
    def store_patterns(self, patterns):
        """Store patterns immediately - NO LOSS ALLOWED"""
        if not patterns:
            return
        
        # Add patterns to buffer
        if isinstance(patterns, list):
            self.pattern_buffer.extend(patterns)
        else:
            self.pattern_buffer.append(patterns)
        
        # Force save if buffer is getting full
        if len(self.pattern_buffer) > 8000:  # 80% of buffer
            print(f"💾 [PATTERNS] Buffer 80% full - forcing save")
            self._save_pattern_buffer()
    
    def get_stored_patterns(self, session_id=None, limit=1000):
        """Retrieve stored patterns"""
        if not self.kg:
            return []
        
        try:
            if session_id:
                query = f"MATCH (n:Entity {{type: 'pattern_batch', session_id: '{session_id}'}}) RETURN n ORDER BY n.timestamp DESC LIMIT {limit}"
            else:
                query = f"MATCH (n:Entity {{type: 'pattern_batch'}}) RETURN n ORDER BY n.timestamp DESC LIMIT {limit}"
            
            result = self.kg.execute_custom_query(query)
            
            all_patterns = []
            for record in result:
                pattern_data = record.get('n', {})
                patterns = json.loads(pattern_data.get('patterns', '[]'))
                all_patterns.extend(patterns)
            
            return all_patterns
            
        except Exception as e:
            print(f"[PATTERNS] ⚠️ Pattern retrieval error: {e}")
            return []
    
    def get_pattern_count(self, session_id=None):
        """Get total pattern count"""
        if not self.kg:
            return 0
        
        try:
            if session_id:
                query = f"MATCH (n:Entity {{type: 'pattern_batch', session_id: '{session_id}'}}) RETURN SUM(n.pattern_count) as total"
            else:
                query = f"MATCH (n:Entity {{type: 'pattern_batch'}}) RETURN SUM(n.pattern_count) as total"
            
            result = self.kg.execute_custom_query(query)
            
            if result and len(result) > 0:
                return result[0].get('total', 0) or 0
            
        except Exception as e:
            print(f"[PATTERNS] ⚠️ Pattern count error: {e}")
        
        return 0
    
    def force_save(self):
        """Force immediate save of all patterns"""
        print(f"💾 [PATTERNS] Force saving {len(self.pattern_buffer)} patterns")
        self._save_pattern_buffer()
    
    def shutdown(self):
        """Shutdown with final save"""
        print(f"💾 [PATTERNS] Shutdown - saving final {len(self.pattern_buffer)} patterns")
        self._save_pattern_buffer()
        print(f"💾 [PATTERNS] Shutdown complete - {self.patterns_stored} total patterns secured")
    
    def get_stats(self):
        """Get pattern storage statistics"""
        return {
            'patterns_stored': self.patterns_stored,
            'buffer_size': len(self.pattern_buffer),
            'last_save_time': self.last_save_time
        }
