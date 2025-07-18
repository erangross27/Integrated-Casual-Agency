"""
Physics Learner Module
Specialized learning for physics concepts and laws
"""

import time
import numpy as np
from typing import Dict, List, Any


class PhysicsLearner:
    """Learns physics concepts from environmental observations"""
    
    def __init__(self, causal_reasoning=None):
        self.physics_concepts = {}
        self.physics_laws = {}
        self.experimental_data = []
        self.causal_reasoning = causal_reasoning
        
        # Physics learning parameters
        self.gravity_constant = 9.8
        self.confidence_threshold = 0.8
    
    def learn_from_observation(self, objects: Dict[str, Any], motion_events: List[Dict[str, Any]]):
        """Learn physics concepts from object observations"""
        
        new_concepts = []
        
        # Learn about gravity
        gravity_concepts = self._learn_gravity_effects(objects)
        new_concepts.extend(gravity_concepts)
        
        # Learn about collisions
        collision_concepts = self._learn_collision_dynamics(objects)
        new_concepts.extend(collision_concepts)
        
        # Learn about motion
        motion_concepts = self._learn_motion_concepts(objects, motion_events)
        new_concepts.extend(motion_concepts)
        
        # Learn about forces
        force_concepts = self._learn_force_concepts(objects)
        new_concepts.extend(force_concepts)
        
        return new_concepts
    
    def _learn_gravity_effects(self, objects: Dict[str, Any]):
        """Learn about gravitational effects"""
        
        concept_name = 'gravitational_acceleration'
        new_concepts = []
        
        if concept_name not in self.physics_concepts:
            self.physics_concepts[concept_name] = {
                'description': 'Objects fall with constant acceleration',
                'evidence': [],
                'confidence': 0.0,
                'measurements': []
            }
            new_concepts.append(concept_name)  # Always discover gravity concept first time
        
        # Always add a basic physics observation to get learning started
        basic_concept = 'basic_physics_observation'
        if basic_concept not in self.physics_concepts and objects:
            self.physics_concepts[basic_concept] = {
                'description': 'Basic physics observations detected',
                'evidence': [{'observation': 'objects_present', 'count': len(objects)}],
                'confidence': 0.3,
                'measurements': []
            }
            new_concepts.append(basic_concept)
        
        concept = self.physics_concepts[concept_name]
        
        # Look for falling objects
        for obj_id, obj_data in objects.items():
            # Skip if obj_data is not a dictionary
            if not isinstance(obj_data, dict):
                continue
                
            velocity = obj_data.get('velocity', [0, 0, 0])
            position = obj_data.get('position', [0, 0, 0])
            
            # Check if object is falling (negative y velocity) or just has any movement
            if velocity[1] < -0.1 or any(abs(v) > 0.01 for v in velocity):
                if concept_name not in self.physics_concepts:
                    new_concepts.append(concept_name)
                    
                evidence = {
                    'object_id': obj_id,
                    'velocity': velocity,
                    'position': position,
                    'acceleration_y': velocity[1],  # Simplified
                    'timestamp': time.time()
                }
                
                concept['evidence'].append(evidence)
                concept['measurements'].append(abs(velocity[1]))
                
                # Update confidence based on consistency with gravity
                expected_acceleration = self.gravity_constant
                measured_acceleration = abs(velocity[1])
                
                if abs(measured_acceleration - expected_acceleration) < 2.0:
                    concept['confidence'] = min(1.0, concept['confidence'] + 0.1)
        
        return new_concepts
    
    def _learn_collision_dynamics(self, objects: Dict[str, Any]):
        """Learn about collision dynamics and momentum conservation"""
        
        concept_name = 'momentum_conservation'
        new_concepts = []
        
        if concept_name not in self.physics_concepts:
            self.physics_concepts[concept_name] = {
                'description': 'Momentum is conserved in collisions',
                'evidence': [],
                'confidence': 0.0,
                'collision_data': []
            }
            new_concepts.append(concept_name)
        
        concept = self.physics_concepts[concept_name]
        
        # Detect potential collisions (simplified)
        obj_list = list(objects.items())
        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                obj1_id, obj1_data = obj_list[i]
                obj2_id, obj2_data = obj_list[j]
                
                if self._objects_close(obj1_data, obj2_data):
                    collision_data = {
                        'object1': obj1_id,
                        'object2': obj2_id,
                        'mass1': obj1_data.get('mass', 1.0),
                        'mass2': obj2_data.get('mass', 1.0),
                        'velocity1': obj1_data.get('velocity', [0, 0, 0]),
                        'velocity2': obj2_data.get('velocity', [0, 0, 0]),
                        'timestamp': time.time()
                    }
                    
                    concept['collision_data'].append(collision_data)
                    concept['confidence'] = min(1.0, concept['confidence'] + 0.05)
        
        return new_concepts
    
    def _learn_motion_concepts(self, objects: Dict[str, Any], motion_events: List[Dict[str, Any]]):
        """Learn concepts about object motion"""
        
        concept_name = 'inertia_principle'
        
        if concept_name not in self.physics_concepts:
            self.physics_concepts[concept_name] = {
                'description': 'Objects at rest stay at rest, objects in motion stay in motion',
                'evidence': [],
                'confidence': 0.0,
                'motion_observations': []
            }
        
        concept = self.physics_concepts[concept_name]
        
        # Analyze motion events for inertia evidence
        for event in motion_events:
            if event['type'] == 'motion_start':
                # Object started moving - evidence against inertia without force
                concept['evidence'].append({
                    'type': 'motion_initiation',
                    'object_id': event['object_id'],
                    'timestamp': event['timestamp']
                })
            
            elif event['type'] == 'motion_stop':
                # Object stopped - evidence for friction/resistance
                concept['evidence'].append({
                    'type': 'motion_cessation',
                    'object_id': event['object_id'],
                    'timestamp': event['timestamp']
                })
                concept['confidence'] = min(1.0, concept['confidence'] + 0.08)
        
        return []  # For now, return empty list
    
    def _learn_force_concepts(self, objects: Dict[str, Any]):
        """Learn about forces and their effects"""
        
        concept_name = 'force_acceleration_relationship'
        
        if concept_name not in self.physics_concepts:
            self.physics_concepts[concept_name] = {
                'description': 'Force equals mass times acceleration (F=ma)',
                'evidence': [],
                'confidence': 0.0,
                'force_measurements': []
            }
        
        concept = self.physics_concepts[concept_name]
        
        # Look for acceleration patterns
        for obj_id, obj_data in objects.items():
            mass = obj_data.get('mass', 1.0)
            acceleration = obj_data.get('acceleration', [0, 0, 0])
            
            if any(abs(a) > 0.1 for a in acceleration):
                force_estimate = [mass * a for a in acceleration]
                
                evidence = {
                    'object_id': obj_id,
                    'mass': mass,
                    'acceleration': acceleration,
                    'estimated_force': force_estimate,
                    'timestamp': time.time()
                }
                
                concept['evidence'].append(evidence)
                concept['force_measurements'].append(np.linalg.norm(force_estimate))
                concept['confidence'] = min(1.0, concept['confidence'] + 0.05)
        
        return []  # For now, return empty list
    
    def _objects_close(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
        """Check if objects are close (potential collision)"""
        
        pos1 = np.array(obj1.get('position', [0, 0, 0]))
        pos2 = np.array(obj2.get('position', [0, 0, 0]))
        
        distance = np.linalg.norm(pos2 - pos1)
        return distance < 1.5  # Close proximity threshold
    
    def generate_physics_hypothesis(self, concept_name: str) -> Dict[str, Any]:
        """Generate hypothesis about physics concept"""
        
        if concept_name not in self.physics_concepts:
            return None
        
        concept = self.physics_concepts[concept_name]
        
        if concept['confidence'] > self.confidence_threshold:
            hypothesis = {
                'type': 'physics_law',
                'concept': concept_name,
                'description': concept['description'],
                'confidence': concept['confidence'],
                'evidence_count': len(concept['evidence']),
                'testable': True,
                'timestamp': time.time()
            }
            
            # Store as physics law if high confidence
            if concept['confidence'] > 0.9:
                self.physics_laws[concept_name] = hypothesis
            
            return hypothesis
        
        return None
    
    def predict_behavior(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict object behavior based on learned physics"""
        
        predictions = {}
        
        # Predict gravitational effects
        if 'gravitational_acceleration' in self.physics_concepts:
            gravity_concept = self.physics_concepts['gravitational_acceleration']
            if gravity_concept['confidence'] > 0.7:
                current_velocity = obj_data.get('velocity', [0, 0, 0])
                predicted_velocity = [
                    current_velocity[0],
                    current_velocity[1] - self.gravity_constant * 0.1,  # dt = 0.1
                    current_velocity[2]
                ]
                predictions['velocity_prediction'] = predicted_velocity
        
        # Predict collision outcomes
        if 'momentum_conservation' in self.physics_concepts:
            momentum_concept = self.physics_concepts['momentum_conservation']
            if momentum_concept['confidence'] > 0.7:
                mass = obj_data.get('mass', 1.0)
                velocity = obj_data.get('velocity', [0, 0, 0])
                momentum = [mass * v for v in velocity]
                predictions['momentum'] = momentum
        
        return predictions
    
    def get_physics_knowledge(self) -> Dict[str, Any]:
        """Get all learned physics knowledge"""
        
        # Get causal models from causal reasoning system if available
        causal_laws = {}
        if self.causal_reasoning:
            causal_models = self.causal_reasoning.get_causal_models()
            # Convert causal models to physics laws format
            for model_id, model in causal_models.items():
                if model.get('confidence', 0) > 0.3:  # Only include confident models
                    law_name = f"causal_law_{model['cause']}_to_{model['effect']}"
                    causal_laws[law_name] = {
                        'type': 'causal_relationship',
                        'cause': model['cause'],
                        'effect': model['effect'],
                        'confidence': model.get('confidence', 0),
                        'strength': model.get('strength', 0),
                        'evidence_count': len(model.get('evidence', []))
                    }
        
        # Combine physics laws with causal laws
        all_laws = self.physics_laws.copy()
        all_laws.update(causal_laws)
        
        return {
            'concepts': self.physics_concepts.copy(),
            'laws': all_laws,
            'experimental_data_count': len(self.experimental_data)
        }
    
    def get_confidence_summary(self) -> Dict[str, float]:
        """Get confidence levels for all physics concepts"""
        
        summary = {}
        for concept_name, concept_data in self.physics_concepts.items():
            summary[concept_name] = concept_data['confidence']
        
        return summary
