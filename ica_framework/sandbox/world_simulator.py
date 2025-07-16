"""
World Simulator for True AGI Learning
Orchestrates the complete learning world for autonomous AGI development
"""

import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from ..utils.logger import ica_logger
from .physics_engine import PhysicsEngine
from .learning_environment import LearningEnvironment


class WorldSimulator:
    """
    Complete world simulation for AGI learning
    
    This simulator provides:
    1. Continuous world simulation
    2. Real-time learning opportunities
    3. Autonomous discovery mechanisms
    4. No pre-programmed knowledge
    5. Emergent behavior emergence
    """
    
    def __init__(self, learning_callback: Optional[Callable] = None):
        self.logger = ica_logger
        self.learning_environment = LearningEnvironment()
        self.learning_callback = learning_callback
        
        # Simulation state
        self.running = False
        self.simulation_thread = None
        self.step_count = 0
        self.start_time = None
        
        # Learning tracking
        self.learning_events = []
        self.discovery_events = []
        self.curiosity_events = []
        
        # Simulation parameters
        self.simulation_speed = 0.1  # Time between steps
        self.auto_generate_events = True
        self.event_generation_probability = 0.1
        
        self.logger.info("World simulator initialized for true AGI learning")
    
    def start(self):
        """Start the world simulation"""
        if self.running:
            self.logger.warning("World simulation already running")
            return
        
        self.running = True
        self.start_time = time.time()
        self.learning_environment.start()
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.logger.info("World simulation started")
    
    def stop(self):
        """Stop the world simulation"""
        if not self.running:
            return
        
        self.running = False
        self.learning_environment.stop()
        
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
        
        self.logger.info("World simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop"""
        self.logger.info("Starting world simulation loop")
        
        while self.running:
            try:
                # Execute simulation step
                learning_opportunity = self.learning_environment.step()
                
                # Process learning opportunity
                if learning_opportunity:
                    self._process_learning_opportunity(learning_opportunity)
                
                # Generate random events for learning
                if self.auto_generate_events and np.random.random() < self.event_generation_probability:
                    self._generate_learning_event()
                
                # Call learning callback if provided
                if self.learning_callback:
                    self.learning_callback(learning_opportunity)
                
                self.step_count += 1
                
                # Sleep for simulation speed
                time.sleep(self.simulation_speed)
                
            except Exception as e:
                self.logger.error(f"Error in simulation loop: {e}")
                time.sleep(1.0)  # Longer sleep on error
    
    def _process_learning_opportunity(self, opportunity: Dict[str, Any]):
        """Process a learning opportunity"""
        
        # Extract learning context
        context = opportunity.get('learning_context', {})
        patterns = context.get('patterns_to_discover', [])
        curiosity_triggers = context.get('curiosity_triggers', [])
        
        # Record learning event
        learning_event = {
            'type': 'learning_opportunity',
            'step': self.step_count,
            'timestamp': time.time(),
            'patterns': patterns,
            'curiosity_triggers': curiosity_triggers,
            'opportunity': opportunity
        }
        
        self.learning_events.append(learning_event)
        
        # Process patterns
        for pattern in patterns:
            self._process_pattern(pattern, opportunity)
        
        # Process curiosity triggers
        for trigger in curiosity_triggers:
            self._process_curiosity_trigger(trigger, opportunity)
    
    def _process_pattern(self, pattern: str, opportunity: Dict[str, Any]):
        """Process a discovered pattern"""
        
        discovery_event = {
            'type': 'pattern_discovery',
            'pattern': pattern,
            'step': self.step_count,
            'timestamp': time.time(),
            'context': opportunity
        }
        
        self.discovery_events.append(discovery_event)
        
        # Add to learning environment
        self.learning_environment.add_discovered_pattern({
            'pattern_type': pattern,
            'discovery_method': 'autonomous_observation',
            'confidence': 0.8,
            'evidence': opportunity['sensory_input']
        })
    
    def _process_curiosity_trigger(self, trigger: str, opportunity: Dict[str, Any]):
        """Process a curiosity trigger"""
        
        curiosity_event = {
            'type': 'curiosity_trigger',
            'trigger': trigger,
            'step': self.step_count,
            'timestamp': time.time(),
            'context': opportunity
        }
        
        self.curiosity_events.append(curiosity_event)
        
        # Generate hypothesis based on curiosity
        hypothesis = self._generate_hypothesis_from_curiosity(trigger, opportunity)
        
        if hypothesis:
            self.learning_environment.add_hypothesis(hypothesis)
    
    def _generate_hypothesis_from_curiosity(self, trigger: str, opportunity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a hypothesis based on curiosity trigger"""
        
        if trigger.startswith('unexpected_collision'):
            return {
                'type': 'physics_law',
                'statement': 'Objects change direction when they collide',
                'testable': True,
                'variables': ['object_velocity_before', 'object_velocity_after', 'collision_angle']
            }
        
        elif trigger.startswith('object_stopping'):
            return {
                'type': 'physics_law',
                'statement': 'Moving objects eventually stop due to friction',
                'testable': True,
                'variables': ['initial_velocity', 'friction_coefficient', 'stopping_time']
            }
        
        elif trigger == 'energy_conservation_question':
            return {
                'type': 'conservation_law',
                'statement': 'Total energy in the system remains constant',
                'testable': True,
                'variables': ['kinetic_energy', 'potential_energy', 'total_energy']
            }
        
        return None
    
    def _generate_learning_event(self):
        """Generate a random learning event"""
        
        event_types = [
            'add_random_object',
            'apply_random_force',
            'change_physics_parameter',
            'create_collision_scenario'
        ]
        
        event_type = np.random.choice(event_types)
        
        if event_type == 'add_random_object':
            self._add_random_object()
        elif event_type == 'apply_random_force':
            self._apply_random_force()
        elif event_type == 'change_physics_parameter':
            self._change_physics_parameter()
        elif event_type == 'create_collision_scenario':
            self._create_collision_scenario()
    
    def _add_random_object(self):
        """Add a random object to the simulation"""
        
        obj_id = f"random_obj_{self.step_count}"
        obj_type = np.random.choice(['sphere', 'cube'])
        mass = np.random.uniform(0.5, 3.0)
        position = np.array([
            np.random.uniform(-5, 5),
            np.random.uniform(2, 8),
            np.random.uniform(-2, 2)
        ])
        
        self.learning_environment.take_action('add_object', 
                                            object_id=obj_id,
                                            type=obj_type,
                                            mass=mass,
                                            position=position)
        
        self.logger.info(f"Added random object {obj_id} for learning")
    
    def _apply_random_force(self):
        """Apply random force to an object"""
        
        objects = list(self.learning_environment.physics_engine.objects.keys())
        if not objects:
            return
        
        obj_id = np.random.choice(objects)
        force = np.array([
            np.random.uniform(-10, 10),
            np.random.uniform(-5, 15),
            np.random.uniform(-5, 5)
        ])
        
        self.learning_environment.take_action('apply_force',
                                            object_id=obj_id,
                                            force=force)
        
        self.logger.info(f"Applied random force {force} to {obj_id} for learning")
    
    def _change_physics_parameter(self):
        """Change a physics parameter"""
        
        parameter = np.random.choice(['gravity', 'friction'])
        
        if parameter == 'gravity':
            new_gravity = np.random.uniform(5, 15)
            self.learning_environment.take_action('change_gravity', gravity=new_gravity)
        elif parameter == 'friction':
            new_friction = np.random.uniform(0.01, 0.3)
            self.learning_environment.take_action('change_friction', friction=new_friction)
        
        self.logger.info(f"Changed {parameter} for learning")
    
    def _create_collision_scenario(self):
        """Create a scenario likely to cause collisions"""
        
        # Add two objects that will collide
        obj1_id = f"collision_obj1_{self.step_count}"
        obj2_id = f"collision_obj2_{self.step_count}"
        
        self.learning_environment.take_action('add_object',
                                            object_id=obj1_id,
                                            type='sphere',
                                            mass=1.0,
                                            position=np.array([0, 5, 0]))
        
        self.learning_environment.take_action('add_object',
                                            object_id=obj2_id,
                                            type='sphere',
                                            mass=1.5,
                                            position=np.array([1, 5, 0]))
        
        # Apply forces to make them collide
        self.learning_environment.take_action('apply_force',
                                            object_id=obj1_id,
                                            force=np.array([5, 0, 0]))
        
        self.learning_environment.take_action('apply_force',
                                            object_id=obj2_id,
                                            force=np.array([-5, 0, 0]))
        
        self.logger.info("Created collision scenario for learning")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        
        runtime = time.time() - self.start_time if self.start_time else 0
        
        stats = {
            'simulation': {
                'running': self.running,
                'steps': self.step_count,
                'runtime_seconds': runtime,
                'steps_per_second': self.step_count / runtime if runtime > 0 else 0
            },
            'learning': {
                'learning_events': len(self.learning_events),
                'discovery_events': len(self.discovery_events),
                'curiosity_events': len(self.curiosity_events),
                'patterns_discovered': len(self.learning_environment.get_discovered_patterns()),
                'hypotheses_generated': len(self.learning_environment.get_hypotheses()),
                'hypotheses_tested': len([h for h in self.learning_environment.get_hypotheses() if h['tested']])
            },
            'environment': self.learning_environment.get_stats()
        }
        
        return stats
    
    def get_recent_learning_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent learning events"""
        return self.learning_events[-count:]
    
    def get_recent_discoveries(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent discoveries"""
        return self.discovery_events[-count:]
    
    def get_recent_curiosity_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent curiosity events"""
        return self.curiosity_events[-count:]
    
    def reset(self):
        """Reset the world simulation"""
        was_running = self.running
        
        if was_running:
            self.stop()
        
        self.learning_environment.reset()
        self.step_count = 0
        self.learning_events.clear()
        self.discovery_events.clear()
        self.curiosity_events.clear()
        
        if was_running:
            self.start()
        
        self.logger.info("World simulation reset")
    
    def set_learning_callback(self, callback: Callable):
        """Set callback for learning events"""
        self.learning_callback = callback
    
    def set_simulation_speed(self, speed: float):
        """Set simulation speed (time between steps)"""
        self.simulation_speed = max(0.01, speed)
    
    def set_auto_generate_events(self, enable: bool, probability: float = 0.1):
        """Enable/disable automatic event generation"""
        self.auto_generate_events = enable
        self.event_generation_probability = probability
    
    def inject_learning_challenge(self, challenge_type: str, **kwargs):
        """Inject a specific learning challenge"""
        
        if challenge_type == 'gravity_change':
            new_gravity = kwargs.get('gravity', 5.0)
            self.learning_environment.take_action('change_gravity', gravity=new_gravity)
            
        elif challenge_type == 'mass_experiment':
            # Create objects with different masses
            for i, mass in enumerate([0.5, 1.0, 2.0, 4.0]):
                obj_id = f"mass_test_{i}"
                self.learning_environment.take_action('add_object',
                                                    object_id=obj_id,
                                                    type='sphere',
                                                    mass=mass,
                                                    position=np.array([i*2, 8, 0]))
        
        elif challenge_type == 'pendulum':
            # Create pendulum-like scenario
            self.learning_environment.take_action('add_object',
                                                object_id='pendulum_bob',
                                                type='sphere',
                                                mass=1.0,
                                                position=np.array([3, 5, 0]))
            
            # Apply initial force to start oscillation
            self.learning_environment.take_action('apply_force',
                                                object_id='pendulum_bob',
                                                force=np.array([-5, 0, 0]))
        
        self.logger.info(f"Injected learning challenge: {challenge_type}")
    
    def get_physics_laws_discoverable(self) -> Dict[str, Any]:
        """Get the physics laws that can be discovered"""
        return self.learning_environment.physics_engine.get_physics_laws()
