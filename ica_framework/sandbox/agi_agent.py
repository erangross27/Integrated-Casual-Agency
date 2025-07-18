"""
Modular AGI Agent - Streamlined and Clean
Uses modular components for maintainable, focused functionality
"""

import time
import threading
from typing import Dict, List, Any, Optional
from ..utils.logger import ica_logger
from ..enhanced_knowledge_graph import EnhancedKnowledgeGraph
from .world_simulator import WorldSimulator
from .agi_modules import (
    MemorySystem, LearningProgress, HypothesisManager, 
    CausalReasoning, SensoryProcessor, AttentionSystem,
    CuriosityEngine, PatternLearner, PhysicsLearner, 
    ExplorationController
)


class AGIAgent:
    """
    Modular AGI Agent - Clean architecture with specialized modules
    
    This agent uses focused modules for:
    - Memory management
    - Learning progress tracking  
    - Hypothesis formation and testing
    - Causal reasoning
    - Sensory processing
    - Attention management
    - Curiosity-driven exploration
    - Pattern recognition
    - Physics learning
    - Exploration control
    """
    
    def __init__(self, world_simulator: WorldSimulator, knowledge_graph: Optional[EnhancedKnowledgeGraph] = None, analytics_logger=None):
        self.logger = ica_logger
        self.world_simulator = world_simulator
        self.running = False
        self.save_thread = None
        
        # Initialize Knowledge Graph
        self._initialize_knowledge_graph(knowledge_graph)
        
        # Initialize all modular components
        self.memory_system = MemorySystem()
        self.learning_progress = LearningProgress(analytics_logger=analytics_logger)
        self.hypothesis_manager = HypothesisManager()
        self.causal_reasoning = CausalReasoning()
        self.sensory_processor = SensoryProcessor()
        self.attention_system = AttentionSystem()
        self.curiosity_engine = CuriosityEngine()
        self.pattern_learner = PatternLearner()
        self.physics_learner = PhysicsLearner()
        self.exploration_controller = ExplorationController()
        
        # Set up world simulation callback
        self.world_simulator.set_learning_callback(self._process_world_feedback)
        
        # Load previous progress
        self._load_previous_state()
        
        self.logger.info("✅ Modular AGI Agent initialized with all components")
    
    def _initialize_knowledge_graph(self, knowledge_graph: Optional[EnhancedKnowledgeGraph]):
        """Initialize knowledge graph with fallback to memory backend"""
        if knowledge_graph:
            self.knowledge_graph = knowledge_graph
        else:
            try:
                self.knowledge_graph = EnhancedKnowledgeGraph(backend='memory')
                if self.knowledge_graph.connect():
                    self.logger.info("✅ Knowledge graph connected (memory backend)")
                else:
                    self.logger.warning("⚠️ Knowledge graph connection failed")
            except Exception as e:
                self.logger.error(f"❌ Knowledge graph error: {e}")
                self.knowledge_graph = EnhancedKnowledgeGraph(backend='memory')
    
    def start_learning(self):
        """Start the autonomous learning process"""
        self.logger.info("🚀 Starting modular AGI learning process")
        self.running = True
        self.world_simulator.start()
        
        # Begin exploration
        self._initiate_exploration()
        
        # Start periodic saving
        if not self.save_thread or not self.save_thread.is_alive():
            self.save_thread = threading.Thread(target=self._periodic_save)
            self.save_thread.daemon = True
            self.save_thread.start()
    
    def stop_learning(self):
        """Stop learning and save all progress"""
        self.logger.info("🛑 Stopping AGI learning process")
        self.running = False
        self._save_all_progress()
        self.world_simulator.stop()
    
    def _process_world_feedback(self, learning_opportunity: Dict[str, Any]):
        """Main feedback processing - coordinates all modules"""
        if not learning_opportunity:
            return
        
        # Ensure learning_opportunity is a dictionary
        if not isinstance(learning_opportunity, dict):
            self.logger.warning(f"⚠️ Received non-dict learning opportunity: {type(learning_opportunity)}")
            return
        
        try:
            # 1. Store in memory system
            self.logger.debug("Step 1: Storing experience")
            self.memory_system.store_experience(learning_opportunity)
            
            # 2. Process sensory input
            self.logger.debug("Step 2: Processing sensory input")
            sensory_input = learning_opportunity.get('sensory_input', {})
            if not isinstance(sensory_input, dict):
                self.logger.error(f"❌ sensory_input is not dict: {type(sensory_input)}")
                return
            processed_sensory = self.sensory_processor.process_sensory_input(sensory_input)
            
            # 3. Update attention system
            self.logger.debug("Step 3: Updating attention system")
            stimuli = self._extract_stimuli(learning_opportunity)
            attention_result = self.attention_system.process_stimuli(stimuli)
            
            # 4. Learn patterns
            self.logger.debug("Step 4: Learning patterns")
            observations = [learning_opportunity]
            patterns_learned = self.pattern_learner.learn_patterns(observations)
            
            # Update learning progress with patterns learned
            if patterns_learned:
                self.learning_progress.update_concepts(len(patterns_learned))
            
            # 5. Learn physics concepts
            self.logger.debug("Step 5: Learning physics concepts")
            # Extract objects from the correct location in the data structure
            objects = learning_opportunity.get('sensory_input', {}).get('raw_physics', {}).get('objects', {})
            motion_events = processed_sensory.get('motion_events', [])
            physics_concepts = self.physics_learner.learn_from_observation(objects, motion_events)
            
            # Update learning progress with physics concepts
            if physics_concepts:
                self.learning_progress.update_concepts(len(physics_concepts))
            
            # 6. Generate curiosity and questions
            self.logger.debug("Step 6: Generating curiosity")
            current_knowledge = self.learning_progress.get_progress_summary()
            curiosity_result = self.curiosity_engine.assess_curiosity(observations, current_knowledge)
            
            # 7. Form and test hypotheses
            self.logger.debug("Step 7: Forming hypotheses")
            if curiosity_result['generated_questions']:
                # Use current and previous learning opportunities for hypothesis generation
                previous_observations = self.memory_system.get_recent_memories(2)
                if len(previous_observations) >= 1:
                    for question in curiosity_result['generated_questions'][:2]:  # Limit questions
                        # Use the most recent observation and current one
                        prev_obs = previous_observations[-1] if previous_observations else learning_opportunity
                        hypothesis = self.hypothesis_manager.generate_hypothesis(prev_obs, learning_opportunity)
                        if hypothesis:
                            self.hypothesis_manager.test_hypothesis(hypothesis, learning_opportunity)
            
            # 8. Update causal reasoning
            self.logger.debug("Step 8: Updating causal reasoning")
            if processed_sensory.get('events'):
                for event in processed_sensory['events']:
                    self.causal_reasoning.observe_event(event)
            
            # 9. Plan exploration
            self.logger.debug("Step 9: Planning exploration")
            if attention_result['primary_focus']:
                current_pos = learning_opportunity.get('agent_position', [0, 0, 0])
                exploration_action = self.exploration_controller.plan_exploration(
                    current_pos, 
                    learning_opportunity,
                    curiosity_result.get('curiosity_targets', [])
                )
            
            # 10. Update learning progress with GPU discoveries
            self.logger.debug("Step 10: Updating learning progress")
            gpu_discoveries = learning_opportunity.get('gpu_discoveries', {})
            if gpu_discoveries:
                self.learning_progress.process_gpu_discoveries(gpu_discoveries)
            
            # 11. Update overall progress
            self.logger.debug("Step 11: Finalizing progress update")
            self.learning_progress.update_metrics({
                'observations_processed': 1,
                'sensory_inputs_processed': 1,
                'learning_cycles': 1
            })
                    
        except Exception as e:
            self.logger.error(f"❌ Error in _process_world_feedback at step: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        self._update_learning_metrics()
    
    def _extract_stimuli(self, learning_opportunity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract stimuli for attention system"""
        stimuli = []
        
        # Extract objects as stimuli
        objects = learning_opportunity.get('objects', {})
        if not isinstance(objects, dict):
            self.logger.warning(f"⚠️ objects is not dict in _extract_stimuli: {type(objects)}")
            return stimuli
            
        for obj_id, obj_data in objects.items():
            if not isinstance(obj_data, dict):
                self.logger.warning(f"⚠️ obj_data for {obj_id} is not dict: {type(obj_data)}")
                continue
                
            stimulus = {
                'id': obj_id,
                'type': obj_data.get('type', 'object'),
                'position': obj_data.get('position', [0, 0, 0]),
                'velocity': obj_data.get('velocity', [0, 0, 0]),
                'saliency': 0.5,  # Base saliency
                'novelty': self._assess_object_novelty(obj_data),
                'urgency': 0.0
            }
            stimuli.append(stimulus)
        
        return stimuli
    
    def _assess_object_novelty(self, obj_data: Dict[str, Any]) -> float:
        """Quick novelty assessment for attention system"""
        if not isinstance(obj_data, dict):
            return 0.0
        obj_type = obj_data.get('type', 'unknown')
        if not self.memory_system.recall_by_type(obj_type):
            return 0.9  # High novelty for new types
        return 0.3  # Lower novelty for familiar types
    
    def _update_learning_metrics(self):
        """Update learning progress metrics"""
        # Get metrics from all modules
        memory_summary = self.memory_system.get_memory_summary()
        hypothesis_summary = self.hypothesis_manager.get_hypothesis_summary()
        physics_knowledge = self.physics_learner.get_physics_knowledge()
        pattern_summary = self.pattern_learner.get_pattern_summary()
        
        # Update learning progress
        progress_update = {
            'concepts_learned': memory_summary['total_memories'],
            'hypotheses_formed': hypothesis_summary.get('active_count', 0) + hypothesis_summary.get('confirmed_count', 0),
            'hypotheses_confirmed': hypothesis_summary.get('confirmed_count', 0),
            'causal_relationships_discovered': len(self.causal_reasoning.causal_relationships),
            'patterns_recognized': pattern_summary.get('high_confidence_patterns', 0),
            'physics_concepts': len(physics_knowledge['concepts'])
        }
        
        self.learning_progress.update_metrics(progress_update)
    
    def _initiate_exploration(self):
        """Start basic exploration behavior"""
        # Add initial exploration goal
        initial_goal = {
            'description': 'Explore environment and discover objects',
            'position': [10, 0, 10],
            'priority': 0.8,
            'estimated_duration': 5.0
        }
        self.exploration_controller.add_goal(initial_goal)
    
    def _load_previous_state(self):
        """Load previous learning state if available"""
        try:
            # This would load from persistent storage
            self.logger.info("Loading previous learning state...")
            # Implementation would depend on your storage system
        except Exception as e:
            self.logger.warning(f"Could not load previous state: {e}")
    
    def _periodic_save(self):
        """Periodic saving of learning progress"""
        while self.running:
            time.sleep(30)  # Save every 30 seconds
            if self.running:
                self._save_all_progress()
    
    def _save_all_progress(self):
        """Save progress from all modules"""
        try:
            # Save to knowledge graph and file system
            progress = self.get_comprehensive_progress()
            
            # Save to knowledge graph
            if self.knowledge_graph:
                self.knowledge_graph.store_learning_progress(progress)
            
            self.logger.info("✅ All progress saved successfully")
        except Exception as e:
            self.logger.error(f"❌ Error saving progress: {e}")
    
    def get_comprehensive_progress(self) -> Dict[str, Any]:
        """Get complete learning progress from all modules"""
        hypothesis_summary = self.hypothesis_manager.get_hypothesis_summary()
        
        return {
            'learning_progress': self.learning_progress.get_progress_summary(),
            'memory_summary': self.memory_system.get_memory_summary(),
            'hypothesis_summary': hypothesis_summary,
            'physics_knowledge': self.physics_learner.get_physics_knowledge(),
            'pattern_summary': self.pattern_learner.get_pattern_summary(),
            'attention_summary': self.attention_system.get_attention_summary(),
            'curiosity_summary': self.curiosity_engine.get_curiosity_summary(),
            'exploration_summary': self.exploration_controller.get_exploration_summary(),
            'causal_relationships': len(self.causal_reasoning.causal_relationships),
            'timestamp': time.time()
        }
    
    def get_current_focus(self) -> Optional[str]:
        """Get current attention focus"""
        return self.attention_system.get_current_focus()
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics"""
        return self.learning_progress.get_progress_summary()
    
    # Simple delegation methods (keep AGI agent thin)
    def set_exploration_rate(self, rate: float):
        """Delegate to exploration controller"""
        self.exploration_controller.exploration_radius = rate * 10.0
        self.curiosity_engine.question_generation_rate = rate
        self.logger.info(f"✅ Exploration rate set to {rate}")
    
    def set_novelty_threshold(self, threshold: float):
        """Delegate to curiosity engine"""
        self.curiosity_engine.novelty_threshold = threshold
        self.attention_system.relevance_threshold = threshold * 0.5
        self.logger.info(f"✅ Novelty threshold set to {threshold}")
    
    def process_gpu_discoveries(self, gpu_results: Dict[str, Any], cycle_count: int):
        """Process GPU discoveries and update all relevant systems"""
        if not gpu_results or 'processed_entities' not in gpu_results:
            return
            
        processed_count = gpu_results.get('processed_entities', 0)
        if processed_count <= 0:
            return
            
        # Store GPU discoveries in memory system
        for i in range(min(processed_count, 50)):  # Limit to prevent memory overflow
            discovery = {
                'type': 'gpu_discovery',
                'cycle': cycle_count,
                'entity_id': i,
                'confidence': gpu_results.get('confidence', 0.8),
                'patterns_found': gpu_results.get('patterns_found', 0),
                'importance': gpu_results.get('importance', 0.8)  # Add importance for consolidation
            }
            
            # Store in short-term memory first
            self.memory_system.store_short_term(discovery)
            
            # Force consolidation for high-confidence discoveries
            if discovery['confidence'] > 0.7:
                # Wait a moment to ensure it's in short-term memory, then consolidate
                if discovery in self.memory_system.short_term_memory:
                    self.memory_system.consolidate_memory(discovery)
                else:
                    # Force consolidation even if not found in deque
                    self.memory_system.store_long_term(discovery.copy())
                    self.memory_system.memory_stats['consolidations'] += 1
                
                # Generate hypothesis from high-confidence discoveries
                if i > 0:  # Need at least one previous discovery for comparison
                    prev_discovery = {
                        'type': 'gpu_discovery',
                        'cycle': cycle_count,
                        'entity_id': i-1,
                        'confidence': gpu_results.get('confidence', 0.8),
                        'patterns_found': gpu_results.get('patterns_found', 0)
                    }
                    hypothesis = self.hypothesis_manager.generate_hypothesis(prev_discovery, discovery)
                    if hypothesis:
                        # Test the hypothesis with current data
                        self.hypothesis_manager.test_hypothesis(hypothesis, discovery)
        
        # Update learning progress with GPU discoveries
        self.learning_progress.process_gpu_discoveries(gpu_results, cycle_count)
        
        # Update learning metrics to get fresh counts from modules
        self._update_learning_metrics()
    
    def _update_learning_metrics(self):
        """Update learning metrics with current counts from all modules"""
        try:
            # Get current counts from hypothesis manager
            hypothesis_summary = self.hypothesis_manager.get_hypothesis_summary()
            active_hypotheses = hypothesis_summary.get('active_count', 0)
            confirmed_hypotheses = hypothesis_summary.get('confirmed_count', 0)
            
            # Update learning progress with actual counts
            if active_hypotheses > 0:
                current_formed = self.learning_progress.progress.get('hypotheses_formed', 0)
                if active_hypotheses > current_formed:
                    new_hypotheses = active_hypotheses - current_formed
                    self.learning_progress.update_hypotheses_formed(new_hypotheses)
            
            if confirmed_hypotheses > 0:
                current_confirmed = self.learning_progress.progress.get('hypotheses_confirmed', 0) 
                if confirmed_hypotheses > current_confirmed:
                    new_confirmed = confirmed_hypotheses - current_confirmed
                    self.learning_progress.update_hypotheses_confirmed(new_confirmed)
                    
        except Exception as e:
            self.logger.debug(f"Learning metrics update failed: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Delegate to comprehensive progress"""
        return self.get_comprehensive_progress()
    
    def inject_curiosity(self, curiosity_boost: float):
        """Delegate to curiosity engine and exploration controller"""
        self.curiosity_engine.novelty_threshold = max(0.1, self.curiosity_engine.novelty_threshold - curiosity_boost)
        exploration_goal = {
            'description': f'Curiosity boost: {curiosity_boost}',
            'position': [5 + curiosity_boost * 10, 0, 5 + curiosity_boost * 10],
            'priority': 0.9,
            'estimated_duration': 3.0
        }
        self.exploration_controller.add_goal(exploration_goal)
        self.logger.info(f"✅ Curiosity injected: {curiosity_boost}")
