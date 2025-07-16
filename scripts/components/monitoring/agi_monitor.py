#!/usr/bin/env python3
"""
AGI Monitor Module
Handles AGI learning progress monitoring and display
"""

import time
import threading
import random


class AGIMonitor:
    """Monitors AGI learning progress and displays statistics"""
    
    def __init__(self, world_simulator, agi_agent, gpu_processor, database_manager):
        self.world_simulator = world_simulator
        self.agi_agent = agi_agent
        self.gpu_processor = gpu_processor
        self.database_manager = database_manager
        
        self.running = False
        self.cycle_count = 0
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start the AGI monitoring thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("[AGI] 👁️ AGI monitoring thread started")
    
    def stop_monitoring(self):
        """Stop the AGI monitoring thread"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.cycle_count += 1
                
                # GPU-accelerated learning processing
                if self.gpu_processor.use_gpu and self.cycle_count % 1 == 0:
                    observation_data = {
                        'cycle': self.cycle_count,
                        'timestamp': time.time(),
                        'type': 'monitoring_observation',
                        'parallel_processing': True
                    }
                    
                    # Process AGI learning with GPU
                    results = self.gpu_processor.process_agi_learning(observation_data)
                    
                    # Store AGI learning to database
                    if results:
                        self.database_manager.store_agi_learning(self.agi_agent)
                
                # Get learning statistics
                world_stats = self.world_simulator.get_learning_statistics()
                agent_summary = self.agi_agent.get_learning_summary()
                
                # Display progress every 10 cycles
                if self.cycle_count % 10 == 0:
                    self._display_progress(world_stats, agent_summary)
                    
                    # Show GPU stats
                    if self.gpu_processor.use_gpu:
                        self.gpu_processor.show_gpu_stats()
                
                # Inject learning challenges periodically
                if self.cycle_count % 25 == 0:
                    self._inject_learning_challenges()
                
                # Adaptive sleep based on learning activity
                learning_activity = world_stats.get('learning', {}).get('learning_events', 0)
                if learning_activity > 50:
                    sleep_time = 2
                elif learning_activity > 20:
                    sleep_time = 3
                else:
                    sleep_time = 5
                
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"[AGI] ⚠️ Error in AGI monitoring: {e}")
                time.sleep(30)
    
    def _display_progress(self, world_stats, agent_summary):
        """Display AGI learning progress"""
        # World simulation stats
        sim_stats = world_stats.get('simulation', {})
        
        # Agent learning progress
        progress = agent_summary.get('learning_progress', {})
        
        concepts_learned = progress.get('concepts_learned', 0)
        hypotheses_formed = progress.get('hypotheses_formed', 0)
        hypotheses_confirmed = progress.get('hypotheses_confirmed', 0)
        causal_relationships = progress.get('causal_relationships_discovered', 0)
        
        # Database readiness indicator
        database_ready = hypotheses_formed > 0 or causal_relationships > 0 or concepts_learned > 15
        db_status = "🔴 Basic learning phase" if not database_ready else "🟢 Database storage ready"
        
        print(f"\n[AGI] 📊 TRUE AGI Learning Progress (Cycle {self.cycle_count}) - {db_status}")
        print(f"[AGI] 🌍 Simulation: {sim_stats.get('steps', 0)} steps, {sim_stats.get('steps_per_second', 0):.1f} steps/sec")
        print(f"[AGI] 🧠 Concepts: {concepts_learned} | Hypotheses: {hypotheses_formed} formed, {hypotheses_confirmed} confirmed | Causal: {causal_relationships}")
        
        # Memory usage and curiosity
        memory = agent_summary.get('memory_usage', {})
        print(f"[AGI] 💾 Memory: ST={memory.get('short_term', 0)}, LT={memory.get('long_term', 0)} | Curiosity: {agent_summary.get('curiosity_level', 0):.2f}")
        
        # Knowledge base size
        kb_size = agent_summary.get('knowledge_base_size', 0)
        causal_models = agent_summary.get('causal_models', 0)
        print(f"[AGI] 📚 Knowledge Base: {kb_size} concepts, {causal_models} causal models")
    
    def _inject_learning_challenges(self):
        """Inject learning challenges to stimulate AGI growth"""
        challenges = [
            ('mass_experiment', {}),
            ('gravity_change', {'gravity': random.uniform(5, 15)}),
            ('pendulum', {}),
        ]
        
        challenge_type, kwargs = random.choice(challenges)
        
        try:
            self.world_simulator.inject_learning_challenge(challenge_type, **kwargs)
            print(f"[AGI] 🎯 Injected learning challenge: {challenge_type}")
            
            # Boost curiosity after challenge
            self.agi_agent.inject_curiosity(0.2)
            
        except Exception as e:
            print(f"[AGI] ⚠️ Error injecting challenge: {e}")
