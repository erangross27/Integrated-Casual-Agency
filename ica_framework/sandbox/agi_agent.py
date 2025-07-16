"""
True AGI Agent for Autonomous Learning
Learns through environmental interaction and discovery with Neo4j integration
"""

import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from ..utils.logger import ica_logger
from ..enhanced_knowledge_graph import EnhancedKnowledgeGraph
from .world_simulator import WorldSimulator


class AGIAgent:
    """
    True AGI Agent that learns through environmental interaction
    
    This agent:
    1. Learns without pre-programmed knowledge
    2. Develops understanding through observation
    3. Forms hypotheses and tests them
    4. Builds causal models from experience
    5. Exhibits curiosity-driven exploration
    """
    
    def __init__(self, world_simulator: WorldSimulator, knowledge_graph: Optional[EnhancedKnowledgeGraph] = None):
        self.logger = ica_logger
        self.world_simulator = world_simulator
        
        # Initialize Enhanced Knowledge Graph with Neo4j backend
        if knowledge_graph:
            self.knowledge_graph = knowledge_graph
        else:
            # Try to load Neo4j configuration
            try:
                import json
                from pathlib import Path
                config_file = Path(__file__).parent.parent.parent / "config/database/neo4j.json"
                
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    
                    # Initialize with Neo4j backend
                    self.knowledge_graph = EnhancedKnowledgeGraph(
                        backend='neo4j',
                        config=config_data['config']
                    )
                    
                    if self.knowledge_graph.connect():
                        self.logger.info("✅ AGI Agent connected to Neo4j knowledge graph")
                    else:
                        self.logger.warning("⚠️ Neo4j connection failed, falling back to memory")
                        self.knowledge_graph = EnhancedKnowledgeGraph(backend='memory')
                else:
                    self.logger.warning("⚠️ No Neo4j config found, using memory backend")
                    self.knowledge_graph = EnhancedKnowledgeGraph(backend='memory')
                    
            except Exception as e:
                self.logger.error(f"❌ Error initializing knowledge graph: {e}")
                self.knowledge_graph = EnhancedKnowledgeGraph(backend='memory')
        
        # Learning state (now stored in knowledge graph)
        self.knowledge_base = {}
        self.causal_models = {}
        self.active_hypotheses = []
        self.tested_hypotheses = []
        
        # Learning control
        self.running = False
        self.save_thread = None
        
        # Memory systems
        self.short_term_memory = deque(maxlen=100)
        self.long_term_memory = []
        self.episodic_memory = []
        
        # Learning metrics
        self.learning_progress = {
            'concepts_learned': 0,
            'hypotheses_formed': 0,
            'hypotheses_confirmed': 0,
            'causal_relationships_discovered': 0,
            'patterns_recognized': 0
        }
        
        # Curiosity system
        self.curiosity_level = 0.5
        self.novelty_threshold = 0.7
        self.exploration_rate = 0.3
        
        # Attention system
        self.attention_focus = None
        self.attention_history = []
        
        # Set up learning callback
        self.world_simulator.set_learning_callback(self._process_world_feedback)
        
        # Load previous learning progress if available
        self.load_learning_progress()
        
        self.logger.info("AGI Agent initialized for true autonomous learning")
    
    def start_learning(self):
        """Start the autonomous learning process"""
        self.logger.info("Starting autonomous learning process")
        self.running = True
        self.world_simulator.start()
        
        # Begin with basic exploration
        self._initiate_exploration()
        
        # Start periodic progress saving
        if self.save_thread is None or not self.save_thread.is_alive():
            self.save_thread = threading.Thread(target=self._periodic_save_progress)
            self.save_thread.daemon = True
            self.save_thread.start()
    
    def stop_learning(self):
        """Stop the learning process and save progress"""
        self.logger.info("Stopping autonomous learning process")
        
        # Stop the learning process
        self.running = False
        
        # Save current progress before stopping
        self.save_learning_progress()
        
        self.world_simulator.stop()
        self.running = False
    
    def _process_world_feedback(self, learning_opportunity: Dict[str, Any]):
        """Process feedback from world simulation"""
        
        if not learning_opportunity:
            return
        
        # Store in short-term memory
        self.short_term_memory.append(learning_opportunity)
        
        # Process sensory input
        sensory_input = learning_opportunity.get('sensory_input', {})
        self._process_sensory_input(sensory_input)
        
        # Process learning context
        learning_context = learning_opportunity.get('learning_context', {})
        self._process_learning_context(learning_context)
        
        # Update attention
        self._update_attention(learning_opportunity)
        
        # Generate new hypotheses if needed
        self._generate_hypotheses(learning_opportunity)
        
        # Test existing hypotheses
        self._test_hypotheses(learning_opportunity)
        
        # Update causal models
        self._update_causal_models(learning_opportunity)
    
    def _process_sensory_input(self, sensory_input: Dict[str, Any]):
        """Process multi-modal sensory input"""
        
        # Visual processing
        if 'visual' in sensory_input:
            self._process_visual_input(sensory_input['visual'])
        
        # Proprioceptive processing
        if 'proprioceptive' in sensory_input:
            self._process_proprioceptive_input(sensory_input['proprioceptive'])
        
        # Temporal processing
        if 'temporal' in sensory_input:
            self._process_temporal_input(sensory_input['temporal'])
    
    def _process_visual_input(self, visual_data: Dict[str, Any]):
        """Process visual sensory input"""
        
        objects = visual_data.get('objects', [])
        
        for obj in objects:
            # Learn about object properties
            self._learn_object_properties(obj)
            
            # Track object behavior
            self._track_object_behavior(obj)
    
    def _learn_object_properties(self, obj: Dict[str, Any]):
        """Learn properties of objects and store in knowledge graph"""
        
        obj_id = obj.get('id', 'unknown')
        
        # Initialize object knowledge
        if obj_id not in self.knowledge_base:
            self.knowledge_base[obj_id] = {
                'type': 'object',
                'properties': {},
                'behaviors': [],
                'relationships': []
            }
        
        # Store object as concept in knowledge graph
        try:
            self.knowledge_graph.add_entity(
                obj_id,
                f"Object {obj_id}",
                {
                    'type': 'physical_object',
                    'mass': obj.get('mass', 0.0),
                    'position': obj.get('position', [0, 0, 0]),
                    'velocity': obj.get('velocity', [0, 0, 0]),
                    'timestamp': time.time()
                }
            )
        except Exception as e:
            self.logger.error(f"❌ Error storing object in knowledge graph: {e}")
        
        # Learn mass concept
        if 'mass' in obj:
            self._learn_mass_concept(obj_id, obj['mass'])
        
        # Learn position concept
        if 'position' in obj:
            self._learn_position_concept(obj_id, obj['position'])
        
        # Learn velocity concept
        if 'velocity' in obj:
            self._learn_velocity_concept(obj_id, obj['velocity'])
    
    def _learn_mass_concept(self, obj_id: str, mass: float):
        """Learn the concept of mass"""
        
        obj_knowledge = self.knowledge_base[obj_id]
        
        # Store mass property
        obj_knowledge['properties']['mass'] = mass
        
        # Learn mass affects behavior
        if 'mass_affects_behavior' not in self.knowledge_base:
            self.knowledge_base['mass_affects_behavior'] = {
                'type': 'concept',
                'description': 'Objects with different masses behave differently',
                'evidence': [],
                'confidence': 0.0
            }
        
        # Add evidence
        self.knowledge_base['mass_affects_behavior']['evidence'].append({
            'object': obj_id,
            'mass': mass,
            'timestamp': time.time()
        })
    
    def _learn_position_concept(self, obj_id: str, position: np.ndarray):
        """Learn the concept of position"""
        
        obj_knowledge = self.knowledge_base[obj_id]
        
        # Store position
        obj_knowledge['properties']['current_position'] = position.tolist()
        
        # Track position history
        if 'position_history' not in obj_knowledge:
            obj_knowledge['position_history'] = []
        
        obj_knowledge['position_history'].append({
            'position': position.tolist(),
            'timestamp': time.time()
        })
        
        # Learn motion concept
        if len(obj_knowledge['position_history']) > 1:
            self._learn_motion_concept(obj_id)
    
    def _learn_motion_concept(self, obj_id: str):
        """Learn concepts related to motion"""
        
        obj_knowledge = self.knowledge_base[obj_id]
        position_history = obj_knowledge['position_history']
        
        if len(position_history) < 2:
            return
        
        # Calculate movement
        prev_pos = np.array(position_history[-2]['position'])
        curr_pos = np.array(position_history[-1]['position'])
        movement = curr_pos - prev_pos
        
        # Learn that objects can move
        if 'objects_can_move' not in self.knowledge_base:
            self.knowledge_base['objects_can_move'] = {
                'type': 'concept',
                'description': 'Objects can change position over time',
                'evidence': [],
                'confidence': 0.0
            }
        
        # Add evidence
        self.knowledge_base['objects_can_move']['evidence'].append({
            'object': obj_id,
            'movement': movement.tolist(),
            'timestamp': time.time()
        })
        
        # Update confidence
        self._update_concept_confidence('objects_can_move')
    
    def _learn_velocity_concept(self, obj_id: str, velocity: np.ndarray):
        """Learn the concept of velocity"""
        
        obj_knowledge = self.knowledge_base[obj_id]
        
        # Store velocity
        obj_knowledge['properties']['current_velocity'] = velocity.tolist()
        
        # Track velocity history
        if 'velocity_history' not in obj_knowledge:
            obj_knowledge['velocity_history'] = []
        
        obj_knowledge['velocity_history'].append({
            'velocity': velocity.tolist(),
            'timestamp': time.time()
        })
        
        # Learn velocity affects position
        if 'velocity_affects_position' not in self.knowledge_base:
            self.knowledge_base['velocity_affects_position'] = {
                'type': 'causal_relationship',
                'description': 'Objects with velocity change position',
                'evidence': [],
                'confidence': 0.0
            }
        
        # Add evidence
        self.knowledge_base['velocity_affects_position']['evidence'].append({
            'object': obj_id,
            'velocity': velocity.tolist(),
            'timestamp': time.time()
        })
        
        # Update confidence
        self._update_concept_confidence('velocity_affects_position')
    
    def _track_object_behavior(self, obj: Dict[str, Any]):
        """Track and learn object behaviors"""
        
        obj_id = obj.get('id', 'unknown')
        
        # Track collision behavior
        if obj.get('collision_detected', False):
            self._learn_collision_behavior(obj_id, obj)
        
        # Track falling behavior
        if 'position' in obj and 'velocity' in obj:
            self._learn_falling_behavior(obj_id, obj)
    
    def _learn_collision_behavior(self, obj_id: str, obj: Dict[str, Any]):
        """Learn about collision behavior"""
        
        # Create collision concept
        if 'collision_concept' not in self.knowledge_base:
            self.knowledge_base['collision_concept'] = {
                'type': 'concept',
                'description': 'Objects can collide with each other',
                'evidence': [],
                'confidence': 0.0
            }
        
        # Add evidence
        self.knowledge_base['collision_concept']['evidence'].append({
            'object': obj_id,
            'collision_data': obj.get('collision_data', {}),
            'timestamp': time.time()
        })
        
        # Learn collision effects
        if 'collision_changes_velocity' not in self.knowledge_base:
            self.knowledge_base['collision_changes_velocity'] = {
                'type': 'causal_relationship',
                'description': 'Collisions change object velocity',
                'evidence': [],
                'confidence': 0.0
            }
        
        # Form hypothesis about collision
        self._form_collision_hypothesis(obj_id, obj)
    
    def _learn_falling_behavior(self, obj_id: str, obj: Dict[str, Any]):
        """Learn about falling behavior"""
        
        position = np.array(obj['position'])
        velocity = np.array(obj['velocity'])
        
        # Check if object is falling (negative y velocity)
        if velocity[1] < -0.1:
            
            # Create falling concept
            if 'falling_concept' not in self.knowledge_base:
                self.knowledge_base['falling_concept'] = {
                    'type': 'concept',
                    'description': 'Objects fall downward',
                    'evidence': [],
                    'confidence': 0.0
                }
            
            # Add evidence
            self.knowledge_base['falling_concept']['evidence'].append({
                'object': obj_id,
                'position': position.tolist(),
                'velocity': velocity.tolist(),
                'timestamp': time.time()
            })
            
            # Learn gravity concept
            self._learn_gravity_concept(obj_id, obj)
    
    def _learn_gravity_concept(self, obj_id: str, obj: Dict[str, Any]):
        """Learn about gravity"""
        
        # Create gravity concept
        if 'gravity_concept' not in self.knowledge_base:
            self.knowledge_base['gravity_concept'] = {
                'type': 'physics_law',
                'description': 'Objects are pulled downward by gravity',
                'evidence': [],
                'confidence': 0.0
            }
        
        # Add evidence
        self.knowledge_base['gravity_concept']['evidence'].append({
            'object': obj_id,
            'falling_behavior': True,
            'timestamp': time.time()
        })
        
        # Update confidence
        self._update_concept_confidence('gravity_concept')
    
    def _process_learning_context(self, learning_context: Dict[str, Any]):
        """Process learning context and opportunities"""
        
        patterns = learning_context.get('patterns_to_discover', [])
        curiosity_triggers = learning_context.get('curiosity_triggers', [])
        
        # Process patterns
        for pattern in patterns:
            self._process_pattern(pattern, learning_context)
        
        # Process curiosity triggers
        for trigger in curiosity_triggers:
            self._process_curiosity_trigger(trigger, learning_context)
    
    def _process_pattern(self, pattern: str, context: Dict[str, Any]):
        """Process a discovered pattern"""
        
        # Update pattern knowledge
        if pattern not in self.knowledge_base:
            self.knowledge_base[pattern] = {
                'type': 'pattern',
                'description': f'Discovered pattern: {pattern}',
                'occurrences': 0,
                'contexts': []
            }
        else:
            # Check if existing entry is a concept, not a pattern
            existing_entry = self.knowledge_base[pattern]
            if existing_entry.get('type') == 'concept':
                # This is a concept, not a pattern - don't process as pattern
                return
            
            # If it's not a proper pattern structure, reinitialize
            if 'occurrences' not in existing_entry or 'contexts' not in existing_entry:
                self.knowledge_base[pattern] = {
                    'type': 'pattern',
                    'description': f'Discovered pattern: {pattern}',
                    'occurrences': 0,
                    'contexts': []
                }
        
        # Record occurrence
        self.knowledge_base[pattern]['occurrences'] += 1
        self.knowledge_base[pattern]['contexts'].append({
            'context': context,
            'timestamp': time.time()
        })
        
        # Update learning progress
        self.learning_progress['patterns_recognized'] += 1
        
        self.logger.info(f"Learned pattern: {pattern}")
    
    def _process_curiosity_trigger(self, trigger: str, context: Dict[str, Any]):
        """Process a curiosity trigger"""
        
        # Increase curiosity level
        self.curiosity_level = min(1.0, self.curiosity_level + 0.1)
        
        # Focus attention on trigger
        self._focus_attention(trigger, context)
        
        # Generate exploration action
        self._generate_exploration_action(trigger, context)
        
        self.logger.info(f"Curiosity triggered: {trigger}")
    
    def _generate_hypotheses(self, learning_opportunity: Dict[str, Any]):
        """Generate new hypotheses from observations and store in knowledge graph"""
        
        # Analyze recent observations
        recent_observations = list(self.short_term_memory)[-5:]
        
        # Look for patterns across observations
        for i in range(len(recent_observations) - 1):
            current = recent_observations[i]
            next_obs = recent_observations[i + 1]
            
            # Generate causal hypothesis
            hypothesis = self._generate_causal_hypothesis(current, next_obs)
            if hypothesis:
                self.active_hypotheses.append(hypothesis)
                self.learning_progress['hypotheses_formed'] += 1
                
                # Store hypothesis in knowledge graph
                try:
                    hypothesis_id = f"hypothesis_{len(self.active_hypotheses)}"
                    self.knowledge_graph.add_entity(
                        hypothesis_id,
                        hypothesis['description'],
                        {
                            'type': 'hypothesis',
                            'confidence': hypothesis.get('confidence', 0.5),
                            'testable': hypothesis.get('testable', True),
                            'tested': hypothesis.get('tested', False),
                            'timestamp': hypothesis.get('timestamp', time.time())
                        }
                    )
                except Exception as e:
                    self.logger.error(f"❌ Error storing hypothesis in knowledge graph: {e}")
                
                self.logger.info(f"Generated hypothesis: {hypothesis['description']}")
    
    def _generate_causal_hypothesis(self, obs1: Dict[str, Any], obs2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate causal hypothesis from two observations"""
        
        # Look for cause-effect relationships
        sensory1 = obs1.get('sensory_input', {})
        sensory2 = obs2.get('sensory_input', {})
        
        # Check for object interactions
        if 'visual' in sensory1 and 'visual' in sensory2:
            objects1 = sensory1['visual'].get('objects', [])
            objects2 = sensory2['visual'].get('objects', [])
            
            # Find common objects
            common_objects = []
            for obj1 in objects1:
                for obj2 in objects2:
                    if obj1.get('id') == obj2.get('id'):
                        common_objects.append((obj1, obj2))
            
            # Generate hypothesis about object behavior
            for obj1, obj2 in common_objects:
                if self._significant_change(obj1, obj2):
                    return {
                        'type': 'causal_hypothesis',
                        'description': f'Object {obj1.get("id")} changed due to some cause',
                        'evidence': [obs1, obs2],
                        'testable': True,
                        'tested': False,
                        'confidence': 0.5,
                        'timestamp': time.time()
                    }
        
        return None
    
    def _significant_change(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
        """Check if object has significant change"""
        
        # Check position change
        if 'position' in obj1 and 'position' in obj2:
            pos1 = np.array(obj1['position'])
            pos2 = np.array(obj2['position'])
            distance = np.linalg.norm(pos2 - pos1)
            
            if distance > 0.5:  # Significant movement
                return True
        
        # Check velocity change
        if 'velocity' in obj1 and 'velocity' in obj2:
            vel1 = np.array(obj1['velocity'])
            vel2 = np.array(obj2['velocity'])
            change = np.linalg.norm(vel2 - vel1)
            
            if change > 1.0:  # Significant velocity change
                return True
        
        return False
    
    def _test_hypotheses(self, learning_opportunity: Dict[str, Any]):
        """Test active hypotheses against new observations and update knowledge graph"""
        
        for hypothesis in self.active_hypotheses:
            if not hypothesis['tested']:
                result = self._test_hypothesis(hypothesis, learning_opportunity)
                if result is not None:
                    hypothesis['tested'] = True
                    hypothesis['test_result'] = result
                    
                    # Update hypothesis in knowledge graph
                    try:
                        hypothesis_id = f"hypothesis_{self.active_hypotheses.index(hypothesis) + 1}"
                        self.knowledge_graph.update_entity(
                            hypothesis_id,
                            {
                                'tested': True,
                                'test_result': result,
                                'confidence': hypothesis.get('confidence', 0.5) * (1.2 if result else 0.8)
                            }
                        )
                    except Exception as e:
                        self.logger.error(f"❌ Error updating hypothesis in knowledge graph: {e}")
                    
                    if result:
                        self.learning_progress['hypotheses_confirmed'] += 1
                        self.logger.info(f"Confirmed hypothesis: {hypothesis['description']}")
                    else:
                        self.logger.info(f"Rejected hypothesis: {hypothesis['description']}")
    
    def _test_hypothesis(self, hypothesis: Dict[str, Any], observation: Dict[str, Any]) -> Optional[bool]:
        """Test a specific hypothesis"""
        
        # Simple pattern matching for now
        # In a real AGI, this would be more sophisticated
        
        if hypothesis['type'] == 'causal_hypothesis':
            # Look for expected patterns in observation
            description = hypothesis['description']
            
            if 'changed due to' in description:
                # Look for evidence of change
                sensory = observation.get('sensory_input', {})
                if 'visual' in sensory:
                    objects = sensory['visual'].get('objects', [])
                    
                    # Check if any object shows the predicted change
                    for obj in objects:
                        if self._object_shows_change(obj, hypothesis):
                            return True
                
                return False
        
        return None
    
    def _object_shows_change(self, obj: Dict[str, Any], hypothesis: Dict[str, Any]) -> bool:
        """Check if object shows expected change from hypothesis"""
        
        # Extract object ID from hypothesis
        description = hypothesis['description']
        
        # Simple pattern matching
        if obj.get('id', '') in description:
            # Check for velocity changes (common in physics)
            if 'velocity' in obj:
                velocity = np.array(obj['velocity'])
                speed = np.linalg.norm(velocity)
                
                if speed > 0.1:  # Object is moving
                    return True
        
        return False
    
    def _update_causal_models(self, learning_opportunity: Dict[str, Any]):
        """Update causal models based on observations and persist to knowledge graph"""
        
        # Build causal relationships
        self._build_causal_relationships(learning_opportunity)
        
        # Update existing models
        self._update_existing_models(learning_opportunity)
        
        # Store causal models in knowledge graph
        try:
            for model_name, model_data in self.causal_models.items():
                # Create relationship in knowledge graph
                if 'cause' in model_data and 'effect' in model_data:
                    self.knowledge_graph.add_relationship(
                        model_data['cause'],
                        model_data['effect'],
                        'CAUSES',
                        {
                            'confidence': model_data.get('confidence', 0.5),
                            'strength': model_data.get('strength', 0.5),
                            'observations': model_data.get('observations', 1)
                        }
                    )
        except Exception as e:
            self.logger.error(f"❌ Error storing causal models in knowledge graph: {e}")
    
    def _build_causal_relationships(self, observation: Dict[str, Any]):
        """Build causal relationships from observations"""
        
        # Look for temporal sequences
        if len(self.short_term_memory) >= 2:
            prev_obs = self.short_term_memory[-2]
            curr_obs = observation
            
            # Find potential causal links
            self._find_causal_links(prev_obs, curr_obs)
    
    def _find_causal_links(self, prev_obs: Dict[str, Any], curr_obs: Dict[str, Any]):
        """Find causal links between observations"""
        
        # Extract events from observations
        prev_events = self._extract_events(prev_obs)
        curr_events = self._extract_events(curr_obs)
        
        # Look for correlations
        for prev_event in prev_events:
            for curr_event in curr_events:
                if self._could_be_causal(prev_event, curr_event):
                    self._record_causal_link(prev_event, curr_event)
    
    def _extract_events(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract events from an observation"""
        
        events = []
        
        # Extract collision events
        sensory = observation.get('sensory_input', {})
        if 'visual' in sensory:
            objects = sensory['visual'].get('objects', [])
            for obj in objects:
                if obj.get('collision_detected', False):
                    events.append({
                        'type': 'collision',
                        'object': obj.get('id'),
                        'timestamp': time.time()
                    })
        
        # Extract force application events
        learning_context = observation.get('learning_context', {})
        if 'force_applied' in learning_context:
            events.append({
                'type': 'force_applied',
                'details': learning_context['force_applied'],
                'timestamp': time.time()
            })
        
        return events
    
    def _could_be_causal(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> bool:
        """Check if two events could be causally related"""
        
        # Temporal ordering
        if event1['timestamp'] >= event2['timestamp']:
            return False
        
        # Event type compatibility
        if event1['type'] == 'force_applied' and event2['type'] == 'collision':
            return True
        
        if event1['type'] == 'collision' and event2['type'] == 'collision':
            return True
        
        return False
    
    def _record_causal_link(self, cause_event: Dict[str, Any], effect_event: Dict[str, Any]):
        """Record a potential causal link"""
        
        link_id = f"{cause_event['type']}_causes_{effect_event['type']}"
        
        if link_id not in self.causal_models:
            self.causal_models[link_id] = {
                'cause': cause_event['type'],
                'effect': effect_event['type'],
                'strength': 0.0,
                'evidence': [],
                'confidence': 0.0
            }
        
        # Add evidence
        self.causal_models[link_id]['evidence'].append({
            'cause_event': cause_event,
            'effect_event': effect_event,
            'timestamp': time.time()
        })
        
        # Update strength
        self.causal_models[link_id]['strength'] += 0.1
        
        # Update confidence
        evidence_count = len(self.causal_models[link_id]['evidence'])
        self.causal_models[link_id]['confidence'] = min(1.0, evidence_count * 0.1)
        
        self.learning_progress['causal_relationships_discovered'] += 1
        
        self.logger.info(f"Recorded causal link: {link_id}")
    
    def _update_existing_models(self, observation: Dict[str, Any]):
        """Update existing causal models"""
        
        # Update model confidence based on new evidence
        for model_id, model in self.causal_models.items():
            if self._model_applies_to_observation(model, observation):
                model['confidence'] = min(1.0, model['confidence'] + 0.05)
    
    def _model_applies_to_observation(self, model: Dict[str, Any], observation: Dict[str, Any]) -> bool:
        """Check if causal model applies to observation"""
        
        # Extract events from observation
        events = self._extract_events(observation)
        
        # Check if model's effect is present
        for event in events:
            if event['type'] == model['effect']:
                return True
        
        return False
    
    def _update_attention(self, learning_opportunity: Dict[str, Any]):
        """Update attention based on learning opportunity"""
        
        # Calculate novelty
        novelty = self._calculate_novelty(learning_opportunity)
        
        # Update attention if novelty is high
        if novelty > self.novelty_threshold:
            self._focus_attention('high_novelty', learning_opportunity)
        
        # Update attention history
        self.attention_history.append({
            'focus': self.attention_focus,
            'novelty': novelty,
            'timestamp': time.time()
        })
        
        # Keep history manageable
        if len(self.attention_history) > 50:
            self.attention_history.pop(0)
    
    def _calculate_novelty(self, observation: Dict[str, Any]) -> float:
        """Calculate novelty of observation"""
        
        # Simple novelty calculation
        # In real AGI, this would be more sophisticated
        
        novelty = 0.0
        
        # Check for new patterns
        learning_context = observation.get('learning_context', {})
        patterns = learning_context.get('patterns_to_discover', [])
        
        for pattern in patterns:
            if pattern not in self.knowledge_base:
                novelty += 0.3
        
        # Check for unexpected events
        curiosity_triggers = learning_context.get('curiosity_triggers', [])
        novelty += len(curiosity_triggers) * 0.2
        
        return min(1.0, novelty)
    
    def _focus_attention(self, focus_type: str, context: Dict[str, Any]):
        """Focus attention on specific aspect"""
        
        self.attention_focus = {
            'type': focus_type,
            'context': context,
            'timestamp': time.time()
        }
        
        self.logger.info(f"Focused attention on: {focus_type}")
    
    def _generate_exploration_action(self, trigger: str, context: Dict[str, Any]):
        """Generate exploration action based on curiosity"""
        
        if np.random.random() < self.exploration_rate:
            
            # Generate random exploration
            actions = [
                'add_random_object',
                'apply_random_force',
                'change_environment'
            ]
            
            action = np.random.choice(actions)
            
            if action == 'add_random_object':
                self._explore_add_object()
            elif action == 'apply_random_force':
                self._explore_apply_force()
            elif action == 'change_environment':
                self._explore_change_environment()
    
    def _explore_add_object(self):
        """Explore by adding an object"""
        
        # Use world simulator to add object
        self.world_simulator.inject_learning_challenge('mass_experiment')
        
        self.logger.info("Exploring: Added object for investigation")
    
    def _explore_apply_force(self):
        """Explore by applying force"""
        
        # Get objects from world simulator
        env = self.world_simulator.learning_environment
        objects = list(env.physics_engine.objects.keys())
        
        if objects:
            obj_id = np.random.choice(objects)
            force = np.array([
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5)
            ])
            
            env.take_action('apply_force', object_id=obj_id, force=force)
            
            self.logger.info(f"Exploring: Applied force to {obj_id}")
    
    def _explore_change_environment(self):
        """Explore by changing environment"""
        
        # Change gravity randomly
        new_gravity = np.random.uniform(5, 15)
        self.world_simulator.inject_learning_challenge('gravity_change', gravity=new_gravity)
        
        self.logger.info(f"Exploring: Changed gravity to {new_gravity}")
    
    def _initiate_exploration(self):
        """Initiate basic exploration"""
        
        # Add some initial objects
        self.world_simulator.inject_learning_challenge('mass_experiment')
        
        # Create collision scenario
        self.world_simulator._create_collision_scenario()
        
        self.logger.info("Initiated basic exploration")
    
    def _update_concept_confidence(self, concept_name: str):
        """Update confidence in a concept"""
        
        if concept_name in self.knowledge_base:
            concept = self.knowledge_base[concept_name]
            evidence_count = len(concept.get('evidence', []))
            
            # Simple confidence calculation
            concept['confidence'] = min(1.0, evidence_count * 0.1)
            
            self.learning_progress['concepts_learned'] += 1
    
    def _form_collision_hypothesis(self, obj_id: str, obj: Dict[str, Any]):
        """Form hypothesis about collision"""
        
        hypothesis = {
            'type': 'physics_hypothesis',
            'description': 'Collisions change object velocities',
            'evidence': [obj],
            'testable': True,
            'tested': False,
            'confidence': 0.6,
            'timestamp': time.time()
        }
        
        self.active_hypotheses.append(hypothesis)
        self.learning_progress['hypotheses_formed'] += 1
        
        self.logger.info("Formed collision hypothesis")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        
        return {
            'learning_progress': self.learning_progress,
            'knowledge_base_size': len(self.knowledge_base),
            'active_hypotheses': len(self.active_hypotheses),
            'tested_hypotheses': len(self.tested_hypotheses),
            'causal_models': len(self.causal_models),
            'curiosity_level': self.curiosity_level,
            'attention_focus': self.attention_focus,
            'memory_usage': {
                'short_term': len(self.short_term_memory),
                'long_term': len(self.long_term_memory),
                'episodic': len(self.episodic_memory)
            }
        }
    
    def get_knowledge_base(self) -> Dict[str, Any]:
        """Get current knowledge base"""
        return self.knowledge_base.copy()
    
    def get_causal_models(self) -> Dict[str, Any]:
        """Get current causal models"""
        return self.causal_models.copy()
    
    def get_active_hypotheses(self) -> List[Dict[str, Any]]:
        """Get currently active hypotheses"""
        return [h for h in self.active_hypotheses if not h['tested']]
    
    def get_confirmed_hypotheses(self) -> List[Dict[str, Any]]:
        """Get confirmed hypotheses"""
        return [h for h in self.active_hypotheses if h['tested'] and h.get('test_result', False)]
    
    def reset_learning(self):
        """Reset learning state"""
        
        self.knowledge_base.clear()
        self.causal_models.clear()
        self.active_hypotheses.clear()
        self.tested_hypotheses.clear()
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.episodic_memory.clear()
        
        self.learning_progress = {
            'concepts_learned': 0,
            'hypotheses_formed': 0,
            'hypotheses_confirmed': 0,
            'causal_relationships_discovered': 0,
            'patterns_recognized': 0
        }
        
        self.curiosity_level = 0.5
        self.attention_focus = None
        self.attention_history.clear()
        
        self.logger.info("Learning state reset")
    
    def set_exploration_rate(self, rate: float):
        """Set exploration rate"""
        self.exploration_rate = max(0.0, min(1.0, rate))
    
    def set_novelty_threshold(self, threshold: float):
        """Set novelty threshold"""
        self.novelty_threshold = max(0.0, min(1.0, threshold))
    
    def inject_curiosity(self, curiosity_boost: float = 0.2):
        """Inject curiosity boost"""
        self.curiosity_level = min(1.0, self.curiosity_level + curiosity_boost)
        self.logger.info(f"Curiosity boosted to {self.curiosity_level}")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights into learning process"""
        
        insights = {
            'most_confident_concepts': [],
            'most_supported_causal_models': [],
            'learning_velocity': 0.0,
            'discovery_rate': 0.0,
            'attention_patterns': []
        }
        
        # Find most confident concepts
        for name, concept in self.knowledge_base.items():
            if concept.get('confidence', 0) > 0.5:
                insights['most_confident_concepts'].append({
                    'name': name,
                    'confidence': concept['confidence'],
                    'type': concept['type']
                })
        
        # Find most supported causal models
        for name, model in self.causal_models.items():
            if model['confidence'] > 0.5:
                insights['most_supported_causal_models'].append({
                    'name': name,
                    'confidence': model['confidence'],
                    'strength': model['strength']
                })
        
        # Calculate learning velocity
        recent_progress = sum(self.learning_progress.values())
        insights['learning_velocity'] = recent_progress / max(1, len(self.short_term_memory))
        
        # Calculate discovery rate
        discoveries = self.learning_progress['concepts_learned'] + self.learning_progress['patterns_recognized']
        insights['discovery_rate'] = discoveries / max(1, self.learning_progress['hypotheses_formed'])
        
        # Analyze attention patterns
        if self.attention_history:
            focus_types = [att['focus']['type'] if att['focus'] else 'unfocused' for att in self.attention_history]
            unique_focuses = list(set(focus_types))
            insights['attention_patterns'] = [(focus, focus_types.count(focus)) for focus in unique_focuses]
        
        return insights
    
    def save_learning_progress(self):
        """Save current learning progress to Neo4j database"""
        try:
            # Save learning progress as a special node
            progress_data = {
                'type': 'learning_progress',
                'session_id': f"session_{int(time.time())}",
                'concepts_learned': self.learning_progress['concepts_learned'],
                'hypotheses_formed': self.learning_progress['hypotheses_formed'],
                'hypotheses_confirmed': self.learning_progress['hypotheses_confirmed'],
                'causal_relationships_discovered': self.learning_progress['causal_relationships_discovered'],
                'patterns_recognized': self.learning_progress['patterns_recognized'],
                'curiosity_level': self.curiosity_level,
                'exploration_rate': self.exploration_rate,
                'novelty_threshold': self.novelty_threshold,
                'knowledge_base_size': len(self.knowledge_base),
                'causal_models_count': len(self.causal_models),
                'active_hypotheses_count': len(self.active_hypotheses),
                'tested_hypotheses_count': len(self.tested_hypotheses),
                'short_term_memory_size': len(self.short_term_memory),
                'long_term_memory_size': len(self.long_term_memory),
                'timestamp': time.time()
            }
            
            self.knowledge_graph.add_entity(
                "learning_progress_current",
                "Current Learning Progress",
                progress_data
            )
            
            # Save active hypotheses
            for i, hypothesis in enumerate(self.active_hypotheses):
                hypothesis_id = f"active_hypothesis_{i}"
                self.knowledge_graph.add_entity(
                    hypothesis_id,
                    hypothesis.get('description', 'Unknown Hypothesis'),
                    {
                        'type': 'active_hypothesis',
                        'confidence': hypothesis.get('confidence', 0.5),
                        'testable': hypothesis.get('testable', True),
                        'tested': hypothesis.get('tested', False),
                        'timestamp': hypothesis.get('timestamp', time.time())
                    }
                )
            
            # Save causal models
            for model_name, model_data in self.causal_models.items():
                model_id = f"causal_model_{model_name}"
                self.knowledge_graph.add_entity(
                    model_id,
                    f"Causal Model: {model_name}",
                    {
                        'type': 'causal_model',
                        'model_name': model_name,
                        'confidence': model_data.get('confidence', 0.5),
                        'strength': model_data.get('strength', 0.5),
                        'observations': model_data.get('observations', 1),
                        'timestamp': time.time()
                    }
                )
            
            self.logger.info("✅ Learning progress saved to Neo4j database")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving learning progress: {e}")
    
    def load_learning_progress(self):
        """Load previous learning progress from Neo4j database"""
        try:
            # Load main progress data
            progress_entity = self.knowledge_graph.get_entity("learning_progress_current")
            
            if progress_entity:
                # Restore learning progress
                self.learning_progress['concepts_learned'] = progress_entity.get('concepts_learned', 0)
                self.learning_progress['hypotheses_formed'] = progress_entity.get('hypotheses_formed', 0)
                self.learning_progress['hypotheses_confirmed'] = progress_entity.get('hypotheses_confirmed', 0)
                self.learning_progress['causal_relationships_discovered'] = progress_entity.get('causal_relationships_discovered', 0)
                self.learning_progress['patterns_recognized'] = progress_entity.get('patterns_recognized', 0)
                
                # Restore system parameters
                self.curiosity_level = progress_entity.get('curiosity_level', 0.5)
                self.exploration_rate = progress_entity.get('exploration_rate', 0.3)
                self.novelty_threshold = progress_entity.get('novelty_threshold', 0.7)
                
                self.logger.info(f"✅ Learning progress restored from database:")
                self.logger.info(f"   • Concepts Learned: {self.learning_progress['concepts_learned']}")
                self.logger.info(f"   • Hypotheses Formed: {self.learning_progress['hypotheses_formed']}")
                self.logger.info(f"   • Causal Relationships: {self.learning_progress['causal_relationships_discovered']}")
                self.logger.info(f"   • Patterns Recognized: {self.learning_progress['patterns_recognized']}")
                
                # Load active hypotheses
                self._load_active_hypotheses()
                
                # Load causal models
                self._load_causal_models()
                
                return True
            else:
                self.logger.info("🆕 No previous learning progress found - starting fresh")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error loading learning progress: {e}")
            return False
    
    def _load_active_hypotheses(self):
        """Load active hypotheses from database"""
        try:
            # Query for active hypotheses
            # This would need to be implemented based on the actual query capabilities
            # For now, we'll implement a basic version
            self.logger.info("🔄 Loading active hypotheses from database...")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading active hypotheses: {e}")
    
    def _load_causal_models(self):
        """Load causal models from database"""
        try:
            # Query for causal models
            # This would need to be implemented based on the actual query capabilities
            # For now, we'll implement a basic version
            self.logger.info("🔄 Loading causal models from database...")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading causal models: {e}")
    
    def _periodic_save_progress(self):
        """Periodically save learning progress"""
        save_interval = 30  # Save every 30 seconds
        last_save = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                if current_time - last_save >= save_interval:
                    self.save_learning_progress()
                    last_save = current_time
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error in periodic save: {e}")
                time.sleep(10)
