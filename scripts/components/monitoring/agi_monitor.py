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
    
    def __init__(self, world_simulator, agi_agent, gpu_processor, database_manager=None):
        self.world_simulator = world_simulator
        self.agi_agent = agi_agent
        self.gpu_processor = gpu_processor
        self.database_manager = database_manager
        
        self.running = False
        self.cycle_count = 0
        self.save_counter = 0
        self.save_interval = 20  # Save every 20 cycles (approximately every 30-40 seconds)
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
                self.save_counter += 1
                
                # Increment analytics learning cycle for epoch advancement
                if hasattr(self, 'database_manager') and self.database_manager and hasattr(self.database_manager, 'analytics_logger'):
                    if self.database_manager.analytics_logger:
                        self.database_manager.analytics_logger.increment_learning_cycle()
                
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
                    
                    # Update AGI agent with GPU-generated discoveries using the new method
                    if results:
                        self.agi_agent.process_gpu_discoveries(results, self.cycle_count)
                    
                    # Store AGI learning to database only periodically
                    if results and self.save_counter >= self.save_interval:
                        self.database_manager.store_learning_state(self.agi_agent, self.gpu_processor)
                        self.save_counter = 0
                
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
        """Display enhanced AGI learning progress with trend analysis"""
        # World simulation stats
        sim_stats = world_stats.get('simulation', {})
        
        # Agent learning progress - fix the path to access nested progress data
        learning_progress_data = agent_summary.get('learning_progress', {})
        progress = learning_progress_data.get('progress', {})
        
        # Get current session learning
        session_concepts = progress.get('concepts_learned', 0)
        session_hypotheses_formed = progress.get('hypotheses_formed', 0)
        session_hypotheses_confirmed = progress.get('hypotheses_confirmed', 0)
        session_causal = progress.get('causal_relationships_discovered', 0)
        
        # Get persistent stats from analytics logger
        persistent_stats = None
        if hasattr(self, 'database_manager') and self.database_manager.analytics_logger:
            persistent_stats = self.database_manager.analytics_logger.get_persistent_stats()
            
        # Use persistent stats for display (total across all sessions)
        if persistent_stats:
            # For all statistics, use persistent stats directly (they all update immediately now)
            concepts_learned = persistent_stats['total_concepts_learned']
            hypotheses_formed = persistent_stats['total_hypotheses_formed']
            hypotheses_confirmed = persistent_stats['total_hypotheses_confirmed']
            causal_relationships = persistent_stats['total_causal_relationships']
        else:
            # Fallback to session stats if persistent stats not available
            concepts_learned = session_concepts
            hypotheses_formed = session_hypotheses_formed
            hypotheses_confirmed = session_hypotheses_confirmed
            causal_relationships = session_causal
        
        # Also check hypothesis manager directly for more accurate counts (for session data only)
        hypothesis_summary = agent_summary.get('hypothesis_summary', {})
        if hypothesis_summary and not persistent_stats:
            # Use hypothesis manager counts if available and no persistent stats
            hypotheses_formed = max(hypotheses_formed, hypothesis_summary.get('active_count', 0))
            hypotheses_confirmed = max(hypotheses_confirmed, hypothesis_summary.get('confirmed_count', 0))
        
        # Also check for direct causal_relationships in agent_summary (fallback for session data)
        if not persistent_stats and causal_relationships == 0:
            causal_relationships = agent_summary.get('causal_relationships', 0)
        
        # Track learning velocity (concepts per cycle)
        if not hasattr(self, 'prev_concepts'):
            self.prev_concepts = concepts_learned
            self.learning_velocity = 0
        else:
            self.learning_velocity = concepts_learned - self.prev_concepts
            self.prev_concepts = concepts_learned
        
        # Database readiness indicator with improved thresholds
        basic_learning = concepts_learned < 100
        intermediate_learning = 100 <= concepts_learned < 1000
        advanced_learning = concepts_learned >= 1000
        
        if basic_learning:
            db_status = "🔴 Basic learning phase"
            phase_icon = "🌱"
        elif intermediate_learning:
            db_status = "🟡 Concept formation active"
            phase_icon = "🌿"
        else:
            db_status = "🟢 Advanced learning phase"
            phase_icon = "🌳"
        
        print(f"\n[AGI] 📊 TRUE AGI Learning Progress (Cycle {self.cycle_count}) - {db_status}")
        print(f"[AGI] 🌍 Simulation: {sim_stats.get('steps', 0)} steps, {sim_stats.get('steps_per_second', 0):.1f} steps/sec")
        
        # Enhanced concept learning display
        velocity_trend = f"(+{self.learning_velocity}/cycle)" if self.learning_velocity > 0 else ""
        
        # Show stats - concepts update immediately, so no session breakdown needed
        if persistent_stats:
            print(f"[AGI] 🧠 Concepts: {concepts_learned} {velocity_trend} | Hypotheses: {hypotheses_formed} formed, {hypotheses_confirmed} confirmed | Causal: {causal_relationships}")
        else:
            print(f"[AGI] 🧠 Concepts: {concepts_learned} {velocity_trend} | Hypotheses: {hypotheses_formed} formed, {hypotheses_confirmed} confirmed | Causal: {causal_relationships}")
        
        # Memory usage and curiosity - fix the paths to access nested data
        memory_summary = agent_summary.get('memory_summary', {})
        
        # Get memory counts from multiple sources for reliability
        memory_usage = memory_summary.get('usage', {})
        short_term = memory_usage.get('short_term', 0)
        long_term = memory_usage.get('long_term', 0)
        
        # Also check direct memory counts if available
        if short_term == 0 and long_term == 0:
            short_term = memory_summary.get('short_term', 0)
            long_term = memory_summary.get('long_term', 0)
        
        curiosity_summary = agent_summary.get('curiosity_summary', {})
        curiosity_level = curiosity_summary.get('average_curiosity_level', 0.0)
        
        print(f"[AGI] 💾 Memory: ST={short_term}, LT={long_term} | Curiosity: {curiosity_level:.2f}")
        
        # Knowledge base size with learning efficiency
        physics_knowledge = agent_summary.get('physics_knowledge', {})
        kb_size = len(physics_knowledge.get('concepts', {}))
        causal_models = len(physics_knowledge.get('laws', {}))
        
        # Calculate learning efficiency (concepts per simulation step)
        total_steps = sim_stats.get('steps', 1)
        learning_efficiency = concepts_learned / max(total_steps, 1) * 100  # concepts per 100 steps
        
        print(f"[AGI] 📚 Knowledge Base: {kb_size} concepts, {causal_models} causal models")
        
        # Add learning phase indicator and efficiency
        if self.cycle_count % 5 == 0:  # Show detailed stats every 5 cycles
            print(f"[AGI] {phase_icon} Learning Phase: {db_status.split(' - ')[0]} | Efficiency: {learning_efficiency:.2f} concepts/100 steps")
            
            # Show learning milestones
            if concepts_learned >= 1000 and not hasattr(self, 'milestone_1000'):
                print(f"[AGI] 🎉 MILESTONE: 1,000 concepts learned! Advanced cognitive development achieved.")
                self.milestone_1000 = True
            elif concepts_learned >= 5000 and not hasattr(self, 'milestone_5000'):
                print(f"[AGI] 🎉 MILESTONE: 5,000 concepts learned! Expert-level pattern recognition achieved.")
                self.milestone_5000 = True
    
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
