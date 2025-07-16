#!/usr/bin/env python3
"""
Learning Coordinator Module
Manages the TRUE AGI learning process
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ..system import SystemUtils

# Setup utilities
flush_print = SystemUtils.flush_print


class LearningCoordinator:
    """Coordinates the TRUE AGI learning process"""
    
    def __init__(self, agi_agent, agi_monitor, gpu_worker):
        self.agi_agent = agi_agent
        self.agi_monitor = agi_monitor
        self.gpu_worker = gpu_worker
        self.learning_active = False
    
    def start_learning_process(self):
        """Start the TRUE AGI learning process"""
        flush_print("[AGI] 🚀 Starting TRUE AGI Learning Process...")
        
        if not self.agi_agent:
            flush_print("[ERROR] ❌ Cannot start learning - no AGI agent")
            return False
        
        # Start the AGI learning system
        self.agi_agent.start_learning()
        
        # Start monitoring
        if self.agi_monitor:
            self.agi_monitor.start_monitoring()
        else:
            flush_print("[WARNING] ⚠️ No AGI monitor - continuing without monitoring")
        
        # Start GPU worker
        if self.gpu_worker:
            self.gpu_worker.start_worker()
        else:
            flush_print("[WARNING] ⚠️ No GPU worker - continuing without GPU processing")
        
        self.learning_active = True
        flush_print("[AGI] ✅ TRUE AGI Learning Process started")
        return True
    
    def stop_learning_process(self):
        """Stop the TRUE AGI learning process"""
        flush_print("[AGI] 🛑 Stopping TRUE AGI Learning Process...")
        
        self.learning_active = False
        
        # Stop monitoring
        if self.agi_monitor:
            self.agi_monitor.stop_monitoring()
        
        # Stop GPU worker
        if self.gpu_worker:
            self.gpu_worker.stop_worker()
        
        # Stop AGI agent
        if self.agi_agent:
            self.agi_agent.stop_learning()
        
        flush_print("[AGI] ✅ TRUE AGI Learning Process stopped")
    
    def is_learning_active(self):
        """Check if learning process is active"""
        return self.learning_active
    
    def get_learning_stats(self):
        """Get learning statistics"""
        stats = {
            'learning_active': self.learning_active,
            'agi_agent_active': self.agi_agent is not None,
            'monitor_active': self.agi_monitor is not None,
            'gpu_worker_active': self.gpu_worker is not None
        }
        
        if self.agi_agent:
            stats['agi_stats'] = {
                'concepts_learned': len(getattr(self.agi_agent, 'knowledge_base', {})),
                'learning_progress': getattr(self.agi_agent, 'learning_progress', {})
            }
        
        return stats
