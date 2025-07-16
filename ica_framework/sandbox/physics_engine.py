"""
Physics Engine for True AGI Learning
Simulates physical interactions that the AGI can observe and learn from
"""

import numpy as np
import random
import time
from typing import Dict, List, Any, Tuple
from ..utils.logger import ica_logger


class PhysicsEngine:
    """
    Real-time physics simulation for AGI learning
    
    This engine simulates basic physical phenomena that an AGI can observe,
    interact with, and learn from. It's designed to provide a continuous
    stream of cause-and-effect relationships for autonomous discovery.
    """
    
    def __init__(self):
        self.logger = ica_logger
        self.time_step = 0.1
        self.objects = {}
        self.forces = {}
        self.interactions = []
        self.history = []
        self.running = False
        
        # Initialize basic physics constants
        self.gravity = 9.8
        self.friction = 0.1
        self.air_resistance = 0.01
        
        # Create initial objects for learning
        self._create_initial_objects()
        
    def _create_initial_objects(self):
        """Create initial objects for the AGI to observe"""
        
        # Create basic objects with different properties
        self.objects = {
            'ball_1': {
                'type': 'sphere',
                'mass': 1.0,
                'position': np.array([0.0, 10.0, 0.0]),
                'velocity': np.array([0.0, 0.0, 0.0]),
                'acceleration': np.array([0.0, 0.0, 0.0]),
                'radius': 0.5,
                'color': 'red',
                'material': 'rubber'
            },
            'ball_2': {
                'type': 'sphere',
                'mass': 2.0,
                'position': np.array([5.0, 10.0, 0.0]),
                'velocity': np.array([0.0, 0.0, 0.0]),
                'acceleration': np.array([0.0, 0.0, 0.0]),
                'radius': 0.7,
                'color': 'blue',
                'material': 'metal'
            },
            'block_1': {
                'type': 'cube',
                'mass': 3.0,
                'position': np.array([2.0, 5.0, 0.0]),
                'velocity': np.array([0.0, 0.0, 0.0]),
                'acceleration': np.array([0.0, 0.0, 0.0]),
                'size': np.array([1.0, 1.0, 1.0]),
                'color': 'green',
                'material': 'wood'
            }
        }
        
        self.logger.info(f"Created {len(self.objects)} initial objects for physics simulation")
    
    def start_simulation(self):
        """Start the physics simulation"""
        self.running = True
        self.logger.info("Physics simulation started")
        
    def stop_simulation(self):
        """Stop the physics simulation"""
        self.running = False
        self.logger.info("Physics simulation stopped")
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one physics simulation step
        
        Returns:
            Dict containing observations from this step
        """
        if not self.running:
            return {}
        
        # Apply forces to all objects
        for obj_id, obj in self.objects.items():
            self._apply_forces(obj_id, obj)
        
        # Update physics for all objects
        for obj_id, obj in self.objects.items():
            self._update_physics(obj_id, obj)
        
        # Check for collisions
        self._check_collisions()
        
        # Create observation
        observation = self._create_observation()
        
        # Store in history
        self.history.append(observation)
        
        return observation
    
    def _apply_forces(self, obj_id: str, obj: Dict[str, Any]):
        """Apply physics forces to an object"""
        
        # Reset acceleration
        obj['acceleration'] = np.array([0.0, 0.0, 0.0])
        
        # Apply gravity
        if obj['position'][1] > 0:  # Above ground
            gravity_force = np.array([0.0, -self.gravity * obj['mass'], 0.0])
            obj['acceleration'] += gravity_force / obj['mass']
        
        # Apply air resistance
        if np.linalg.norm(obj['velocity']) > 0:
            resistance_force = -self.air_resistance * obj['velocity'] * np.linalg.norm(obj['velocity'])
            obj['acceleration'] += resistance_force / obj['mass']
        
        # Apply friction if on ground
        if obj['position'][1] <= 0 and np.linalg.norm(obj['velocity'][:2]) > 0:
            friction_force = -self.friction * obj['velocity'][:2] * self.gravity
            obj['acceleration'][:2] += friction_force / obj['mass']
    
    def _update_physics(self, obj_id: str, obj: Dict[str, Any]):
        """Update object physics using basic integration"""
        
        # Update velocity
        obj['velocity'] += obj['acceleration'] * self.time_step
        
        # Update position
        obj['position'] += obj['velocity'] * self.time_step
        
        # Ground collision
        if obj['position'][1] <= 0:
            obj['position'][1] = 0
            obj['velocity'][1] = -obj['velocity'][1] * 0.8  # Bounce with energy loss
            
            # Stop if velocity is very small
            if abs(obj['velocity'][1]) < 0.1:
                obj['velocity'][1] = 0
    
    def _check_collisions(self):
        """Check for collisions between objects"""
        
        objects_list = list(self.objects.items())
        
        for i in range(len(objects_list)):
            for j in range(i + 1, len(objects_list)):
                obj1_id, obj1 = objects_list[i]
                obj2_id, obj2 = objects_list[j]
                
                # Simple collision detection
                distance = np.linalg.norm(obj1['position'] - obj2['position'])
                min_distance = self._get_collision_radius(obj1) + self._get_collision_radius(obj2)
                
                if distance < min_distance:
                    self._handle_collision(obj1_id, obj1, obj2_id, obj2)
    
    def _get_collision_radius(self, obj: Dict[str, Any]) -> float:
        """Get collision radius for an object"""
        if obj['type'] == 'sphere':
            return obj['radius']
        elif obj['type'] == 'cube':
            return np.linalg.norm(obj['size']) / 2
        return 0.5
    
    def _handle_collision(self, obj1_id: str, obj1: Dict[str, Any], 
                         obj2_id: str, obj2: Dict[str, Any]):
        """Handle collision between two objects"""
        
        # Simple elastic collision
        m1, m2 = obj1['mass'], obj2['mass']
        v1, v2 = obj1['velocity'], obj2['velocity']
        
        # Calculate collision normal
        normal = obj2['position'] - obj1['position']
        normal = normal / np.linalg.norm(normal)
        
        # Calculate relative velocity
        relative_velocity = v1 - v2
        velocity_along_normal = np.dot(relative_velocity, normal)
        
        # Don't resolve if velocities are separating
        if velocity_along_normal > 0:
            return
        
        # Calculate collision impulse
        impulse = 2 * velocity_along_normal / (m1 + m2)
        
        # Update velocities
        obj1['velocity'] -= impulse * m2 * normal
        obj2['velocity'] += impulse * m1 * normal
        
        # Record collision
        self.interactions.append({
            'type': 'collision',
            'objects': [obj1_id, obj2_id],
            'time': time.time(),
            'impulse': impulse,
            'normal': normal.tolist()
        })
    
    def _create_observation(self) -> Dict[str, Any]:
        """Create observation from current physics state"""
        
        observation = {
            'time': time.time(),
            'objects': {},
            'interactions': self.interactions[-5:],  # Last 5 interactions
            'forces': {
                'gravity': self.gravity,
                'friction': self.friction,
                'air_resistance': self.air_resistance
            }
        }
        
        # Add object states
        for obj_id, obj in self.objects.items():
            observation['objects'][obj_id] = {
                'type': obj['type'],
                'mass': obj['mass'],
                'position': obj['position'].tolist(),
                'velocity': obj['velocity'].tolist(),
                'acceleration': obj['acceleration'].tolist(),
                'kinetic_energy': 0.5 * obj['mass'] * np.linalg.norm(obj['velocity'])**2,
                'potential_energy': obj['mass'] * self.gravity * obj['position'][1],
                'material': obj['material'],
                'color': obj['color']
            }
        
        return observation
    
    def apply_force(self, obj_id: str, force: np.ndarray):
        """Apply external force to an object"""
        
        if obj_id in self.objects:
            obj = self.objects[obj_id]
            obj['acceleration'] += force / obj['mass']
            
            # Record force application
            self.interactions.append({
                'type': 'applied_force',
                'object': obj_id,
                'force': force.tolist(),
                'time': time.time()
            })
    
    def add_object(self, obj_id: str, obj_type: str, mass: float, 
                   position: np.ndarray, **kwargs):
        """Add a new object to the simulation"""
        
        new_obj = {
            'type': obj_type,
            'mass': mass,
            'position': position.copy(),
            'velocity': np.array([0.0, 0.0, 0.0]),
            'acceleration': np.array([0.0, 0.0, 0.0]),
            'material': kwargs.get('material', 'default'),
            'color': kwargs.get('color', 'white')
        }
        
        # Add type-specific properties
        if obj_type == 'sphere':
            new_obj['radius'] = kwargs.get('radius', 0.5)
        elif obj_type == 'cube':
            new_obj['size'] = kwargs.get('size', np.array([1.0, 1.0, 1.0]))
        
        self.objects[obj_id] = new_obj
        self.logger.info(f"Added new object {obj_id} of type {obj_type}")
    
    def remove_object(self, obj_id: str):
        """Remove an object from the simulation"""
        
        if obj_id in self.objects:
            del self.objects[obj_id]
            self.logger.info(f"Removed object {obj_id}")
    
    def get_observation_history(self, steps: int = 10) -> List[Dict[str, Any]]:
        """Get the last N observations"""
        return self.history[-steps:]
    
    def reset(self):
        """Reset the simulation to initial state"""
        self.objects.clear()
        self.interactions.clear()
        self.history.clear()
        self._create_initial_objects()
        self.logger.info("Physics simulation reset")
    
    def get_system_energy(self) -> float:
        """Calculate total system energy"""
        
        total_energy = 0.0
        
        for obj in self.objects.values():
            # Kinetic energy
            kinetic = 0.5 * obj['mass'] * np.linalg.norm(obj['velocity'])**2
            
            # Potential energy
            potential = obj['mass'] * self.gravity * obj['position'][1]
            
            total_energy += kinetic + potential
        
        return total_energy
    
    def get_physics_laws(self) -> Dict[str, Any]:
        """Return discoverable physics laws in the simulation"""
        
        return {
            'conservation_of_energy': {
                'description': 'Total energy is conserved in isolated systems',
                'formula': 'E_total = KE + PE = constant'
            },
            'newtons_second_law': {
                'description': 'Force equals mass times acceleration',
                'formula': 'F = ma'
            },
            'conservation_of_momentum': {
                'description': 'Total momentum is conserved in collisions',
                'formula': 'p_total = m1*v1 + m2*v2 = constant'
            },
            'gravity': {
                'description': 'Objects accelerate downward due to gravity',
                'formula': 'F_gravity = mg'
            }
        }
