"""
Learning Environment for True AGI
Provides a controlled environment where AGI can learn through observation and interaction
"""

import numpy as np
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from ..utils.logger import ica_logger
from .physics_engine import PhysicsEngine


class LearningEnvironment:
    """
    Environment for autonomous AGI learning
    
    This environment provides:
    1. Continuous sensory input from physics simulation
    2. Ability to take actions and observe consequences
    3. No pre-programmed knowledge - pure discovery learning
    4. Emergent pattern recognition opportunities
    """
    
    def __init__(self):
        self.logger = ica_logger
        self.physics_engine = PhysicsEngine()
        self.step_count = 0
        self.learning_history = []
        self.discovered_patterns = []
        self.hypotheses = []
        self.running = False
        
        # AGI sensory capabilities
        self.sensors = {
            'vision': True,
            'position': True,
            'velocity': True,
            'acceleration': True,
            'force': True,
            'energy': True,
            'material': True,
            'collision': True
        }
        
        # AGI action capabilities
        self.actions = {
            'apply_force': True,
            'add_object': True,
            'remove_object': True,
            'change_gravity': True,
            'change_friction': True
        }
        
        self.logger.info("Learning environment initialized for true AGI learning")
    
    def start(self):
        """Start the learning environment"""
        self.running = True
        self.physics_engine.start_simulation()
        self.logger.info("Learning environment started")
    
    def stop(self):
        """Stop the learning environment"""
        self.running = False
        self.physics_engine.stop_simulation()
        self.logger.info("Learning environment stopped")
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one learning step
        
        Returns:
            Complete observation for the AGI to process
        """
        if not self.running:
            return {}
        
        try:
            # Get physics observation
            physics_obs = self.physics_engine.step()
            
            # Process through sensors
            sensory_input = self._process_sensory_input(physics_obs)
            
            # Create learning opportunity
            learning_opportunity = self._create_learning_opportunity(sensory_input)
            
            # Validate that learning_opportunity is a dict
            if not isinstance(learning_opportunity, dict):
                self.logger.error(f"❌ _create_learning_opportunity returned {type(learning_opportunity)}: {learning_opportunity}")
                return {}
            
            # Store in history
            self.learning_history.append(learning_opportunity)
            self.step_count += 1
            
            return learning_opportunity
            
        except Exception as e:
            self.logger.error(f"❌ Error in learning environment step: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
        
        return learning_opportunity
    
    def _process_sensory_input(self, physics_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw physics into sensory input"""
        
        sensory_input = {
            'timestamp': time.time(),
            'step': self.step_count,
            'raw_physics': physics_obs,
            'processed_sensors': {}
        }
        
        # Vision sensor - simplified visual representation
        if self.sensors['vision']:
            sensory_input['processed_sensors']['vision'] = self._process_vision(physics_obs)
        
        # Position sensor
        if self.sensors['position']:
            sensory_input['processed_sensors']['position'] = self._process_position(physics_obs)
        
        # Velocity sensor
        if self.sensors['velocity']:
            sensory_input['processed_sensors']['velocity'] = self._process_velocity(physics_obs)
        
        # Acceleration sensor
        if self.sensors['acceleration']:
            sensory_input['processed_sensors']['acceleration'] = self._process_acceleration(physics_obs)
        
        # Force sensor
        if self.sensors['force']:
            sensory_input['processed_sensors']['force'] = self._process_force(physics_obs)
        
        # Energy sensor
        if self.sensors['energy']:
            sensory_input['processed_sensors']['energy'] = self._process_energy(physics_obs)
        
        # Collision sensor
        if self.sensors['collision']:
            sensory_input['processed_sensors']['collision'] = self._process_collision(physics_obs)
        
        return sensory_input
    
    def _process_vision(self, physics_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process visual information"""
        visual_data = {}
        
        for obj_id, obj in physics_obs.get('objects', {}).items():
            visual_data[obj_id] = {
                'color': obj['color'],
                'type': obj['type'],
                'position': obj['position'],
                'visible': obj['position'][1] >= 0,  # Above ground
                'size': self._estimate_visual_size(obj)
            }
        
        return visual_data
    
    def _estimate_visual_size(self, obj: Dict[str, Any]) -> float:
        """Estimate visual size of object"""
        if obj['type'] == 'sphere':
            return obj.get('radius', 0.5) * 2
        elif obj['type'] == 'cube':
            return max(obj.get('size', [1.0, 1.0, 1.0]))
        return 1.0
    
    def _process_position(self, physics_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process position information"""
        position_data = {}
        
        for obj_id, obj in physics_obs.get('objects', {}).items():
            position_data[obj_id] = {
                'position': obj['position'],
                'distance_from_origin': np.linalg.norm(obj['position']),
                'height': obj['position'][1],
                'on_ground': obj['position'][1] <= 0.1
            }
        
        return position_data
    
    def _process_velocity(self, physics_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process velocity information"""
        velocity_data = {}
        
        for obj_id, obj in physics_obs.get('objects', {}).items():
            velocity_data[obj_id] = {
                'velocity': obj['velocity'],
                'speed': np.linalg.norm(obj['velocity']),
                'direction': obj['velocity'] / (np.linalg.norm(obj['velocity']) + 1e-8),
                'moving': np.linalg.norm(obj['velocity']) > 0.1
            }
        
        return velocity_data
    
    def _process_acceleration(self, physics_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process acceleration information"""
        acceleration_data = {}
        
        for obj_id, obj in physics_obs.get('objects', {}).items():
            acceleration_data[obj_id] = {
                'acceleration': obj['acceleration'],
                'magnitude': np.linalg.norm(obj['acceleration']),
                'accelerating': np.linalg.norm(obj['acceleration']) > 0.1
            }
        
        return acceleration_data
    
    def _process_force(self, physics_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process force information"""
        force_data = {
            'gravity': physics_obs.get('forces', {}).get('gravity', 9.8),
            'friction': physics_obs.get('forces', {}).get('friction', 0.1),
            'air_resistance': physics_obs.get('forces', {}).get('air_resistance', 0.01)
        }
        
        return force_data
    
    def _process_energy(self, physics_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process energy information"""
        energy_data = {}
        
        total_kinetic = 0.0
        total_potential = 0.0
        
        for obj_id, obj in physics_obs.get('objects', {}).items():
            kinetic = obj['kinetic_energy']
            potential = obj['potential_energy']
            
            energy_data[obj_id] = {
                'kinetic_energy': kinetic,
                'potential_energy': potential,
                'total_energy': kinetic + potential
            }
            
            total_kinetic += kinetic
            total_potential += potential
        
        energy_data['system_totals'] = {
            'total_kinetic': total_kinetic,
            'total_potential': total_potential,
            'total_energy': total_kinetic + total_potential
        }
        
        return energy_data
    
    def _process_collision(self, physics_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process collision information"""
        collision_data = {
            'recent_collisions': [],
            'collision_count': 0
        }
        
        for interaction in physics_obs.get('interactions', []):
            if interaction['type'] == 'collision':
                collision_data['recent_collisions'].append({
                    'objects': interaction['objects'],
                    'impulse': interaction['impulse'],
                    'normal': interaction['normal']
                })
                collision_data['collision_count'] += 1
        
        return collision_data
    
    def _create_learning_opportunity(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Create a learning opportunity from sensory input"""
        
        learning_opportunity = {
            'sensory_input': sensory_input,
            'learning_context': {
                'step': self.step_count,
                'timestamp': time.time(),
                'available_actions': list(self.actions.keys()),
                'patterns_to_discover': self._identify_patterns(sensory_input),
                'curiosity_triggers': self._identify_curiosity_triggers(sensory_input)
            },
            'discovery_hints': self._generate_discovery_hints(sensory_input)
        }
        
        return learning_opportunity
    
    def _identify_patterns(self, sensory_input: Dict[str, Any]) -> List[str]:
        """Identify patterns the AGI could discover"""
        patterns = []
        
        # Check for falling objects
        position_data = sensory_input['processed_sensors'].get('position', {})
        velocity_data = sensory_input['processed_sensors'].get('velocity', {})
        
        for obj_id in position_data:
            if (position_data[obj_id]['height'] > 0 and 
                velocity_data.get(obj_id, {}).get('velocity', [0,0,0])[1] < 0):
                patterns.append(f"falling_object_{obj_id}")
        
        # Check for collisions
        collision_data = sensory_input['processed_sensors'].get('collision', {})
        if collision_data.get('collision_count', 0) > 0:
            patterns.append("collision_event")
        
        # Check for energy changes
        energy_data = sensory_input['processed_sensors'].get('energy', {})
        if len(self.learning_history) > 0:
            prev_energy = self.learning_history[-1]['sensory_input']['processed_sensors'].get('energy', {})
            if prev_energy:
                current_total = energy_data.get('system_totals', {}).get('total_energy', 0)
                prev_total = prev_energy.get('system_totals', {}).get('total_energy', 0)
                if abs(current_total - prev_total) > 0.1:
                    patterns.append("energy_change")
        
        return patterns
    
    def _identify_curiosity_triggers(self, sensory_input: Dict[str, Any]) -> List[str]:
        """Identify things that should trigger curiosity"""
        triggers = []
        
        # Unexpected events
        collision_data = sensory_input['processed_sensors'].get('collision', {})
        if collision_data.get('collision_count', 0) > 0:
            triggers.append("unexpected_collision")
        
        # Objects stopping
        velocity_data = sensory_input['processed_sensors'].get('velocity', {})
        for obj_id, data in velocity_data.items():
            if data['speed'] < 0.1 and data['speed'] > 0:
                triggers.append(f"object_stopping_{obj_id}")
        
        # Energy conservation violations (shouldn't happen in good physics)
        energy_data = sensory_input['processed_sensors'].get('energy', {})
        if len(self.learning_history) > 5:
            # Check if energy is being conserved
            recent_energies = [h['sensory_input']['processed_sensors'].get('energy', {}).get('system_totals', {}).get('total_energy', 0) 
                             for h in self.learning_history[-5:]]
            if len(recent_energies) > 1:
                energy_variance = np.var(recent_energies)
                if energy_variance > 1.0:  # High variance suggests investigation needed
                    triggers.append("energy_conservation_question")
        
        return triggers
    
    def _generate_discovery_hints(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hints for what the AGI might discover"""
        
        hints = {
            'physics_laws': [],
            'relationships': [],
            'invariants': []
        }
        
        # Hint at gravity
        position_data = sensory_input['processed_sensors'].get('position', {})
        velocity_data = sensory_input['processed_sensors'].get('velocity', {})
        
        falling_objects = 0
        for obj_id in position_data:
            if (position_data[obj_id]['height'] > 0 and 
                velocity_data.get(obj_id, {}).get('velocity', [0,0,0])[1] < 0):
                falling_objects += 1
        
        if falling_objects > 0:
            hints['physics_laws'].append("downward_acceleration_pattern")
        
        # Hint at conservation of energy
        energy_data = sensory_input['processed_sensors'].get('energy', {})
        if energy_data:
            hints['invariants'].append("total_energy_conservation")
        
        # Hint at mass-acceleration relationship
        acceleration_data = sensory_input['processed_sensors'].get('acceleration', {})
        if acceleration_data:
            hints['relationships'].append("mass_acceleration_relationship")
        
        return hints
    
    def take_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Allow AGI to take action in the environment
        
        Args:
            action: Type of action to take
            **kwargs: Action parameters
            
        Returns:
            Result of the action
        """
        if not self.actions.get(action, False):
            return {'success': False, 'error': f'Action {action} not available'}
        
        result = {'success': True, 'action': action, 'parameters': kwargs}
        
        try:
            if action == 'apply_force':
                obj_id = kwargs.get('object_id')
                force = np.array(kwargs.get('force', [0, 0, 0]))
                self.physics_engine.apply_force(obj_id, force)
                result['effect'] = f'Applied force {force} to {obj_id}'
                
            elif action == 'add_object':
                obj_id = kwargs.get('object_id', f'obj_{self.step_count}')
                obj_type = kwargs.get('type', 'sphere')
                mass = kwargs.get('mass', 1.0)
                position = np.array(kwargs.get('position', [0, 5, 0]))
                self.physics_engine.add_object(obj_id, obj_type, mass, position, **kwargs)
                result['effect'] = f'Added {obj_type} {obj_id} at {position}'
                
            elif action == 'remove_object':
                obj_id = kwargs.get('object_id')
                self.physics_engine.remove_object(obj_id)
                result['effect'] = f'Removed object {obj_id}'
                
            elif action == 'change_gravity':
                new_gravity = kwargs.get('gravity', 9.8)
                self.physics_engine.gravity = new_gravity
                result['effect'] = f'Changed gravity to {new_gravity}'
                
            elif action == 'change_friction':
                new_friction = kwargs.get('friction', 0.1)
                self.physics_engine.friction = new_friction
                result['effect'] = f'Changed friction to {new_friction}'
                
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def get_learning_history(self, steps: int = 10) -> List[Dict[str, Any]]:
        """Get recent learning history"""
        return self.learning_history[-steps:]
    
    def get_discovered_patterns(self) -> List[Dict[str, Any]]:
        """Get patterns discovered by the AGI"""
        return self.discovered_patterns
    
    def add_discovered_pattern(self, pattern: Dict[str, Any]):
        """Add a pattern discovered by the AGI"""
        self.discovered_patterns.append({
            'pattern': pattern,
            'discovery_time': time.time(),
            'step': self.step_count
        })
        self.logger.info(f"AGI discovered pattern: {pattern}")
    
    def get_hypotheses(self) -> List[Dict[str, Any]]:
        """Get AGI hypotheses"""
        return self.hypotheses
    
    def add_hypothesis(self, hypothesis: Dict[str, Any]):
        """Add a hypothesis generated by the AGI"""
        self.hypotheses.append({
            'hypothesis': hypothesis,
            'generation_time': time.time(),
            'step': self.step_count,
            'tested': False,
            'result': None
        })
        self.logger.info(f"AGI generated hypothesis: {hypothesis}")
    
    def test_hypothesis(self, hypothesis_id: int, test_result: Dict[str, Any]):
        """Record the result of testing a hypothesis"""
        if 0 <= hypothesis_id < len(self.hypotheses):
            self.hypotheses[hypothesis_id]['tested'] = True
            self.hypotheses[hypothesis_id]['result'] = test_result
            self.hypotheses[hypothesis_id]['test_time'] = time.time()
            self.logger.info(f"AGI tested hypothesis {hypothesis_id}: {test_result}")
    
    def reset(self):
        """Reset the learning environment"""
        self.physics_engine.reset()
        self.step_count = 0
        self.learning_history.clear()
        self.discovered_patterns.clear()
        self.hypotheses.clear()
        self.logger.info("Learning environment reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'steps': self.step_count,
            'patterns_discovered': len(self.discovered_patterns),
            'hypotheses_generated': len(self.hypotheses),
            'hypotheses_tested': len([h for h in self.hypotheses if h['tested']]),
            'objects_in_simulation': len(self.physics_engine.objects),
            'total_system_energy': self.physics_engine.get_system_energy(),
            'recent_interactions': len(self.physics_engine.interactions[-10:])
        }
