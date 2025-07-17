"""
Sensory Processor Module
Handles processing of sensory input and environmental observations
"""

import time
import numpy as np
from typing import Dict, List, Any


class SensoryProcessor:
    """Processes sensory input from the environment"""
    
    def __init__(self):
        self.object_tracker = {}
        self.motion_history = {}
        self.collision_events = []
        
        # Processing parameters
        self.position_threshold = 0.1
        self.velocity_threshold = 0.05
    
    def process_sensory_input(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process multi-modal sensory input"""
        
        processed_data = {
            'objects': {},
            'motion_events': [],
            'collision_events': [],
            'environmental_changes': []
        }
        
        # Process visual input
        if 'visual' in sensory_input:
            processed_data['objects'] = self._process_visual_input(sensory_input['visual'])
        
        # Process physics data
        if 'raw_physics' in sensory_input and 'objects' in sensory_input['raw_physics']:
            self._merge_physics_data(processed_data['objects'], sensory_input['raw_physics']['objects'])
        
        # Detect motion events
        processed_data['motion_events'] = self._detect_motion_events(processed_data['objects'])
        
        # Detect collision events
        processed_data['collision_events'] = self._detect_collision_events(processed_data['objects'])
        
        return processed_data
    
    def _process_visual_input(self, visual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process visual sensory input"""
        
        processed_objects = {}
        
        for obj_id, obj_data in visual_data.items():
            processed_objects[obj_id] = {
                'id': obj_id,
                'position': obj_data.get('position', [0, 0, 0]),
                'velocity': obj_data.get('velocity', [0, 0, 0]),
                'properties': self._extract_object_properties(obj_data)
            }
            
            # Update object tracker
            self._update_object_tracker(obj_id, processed_objects[obj_id])
        
        return processed_objects
    
    def _merge_physics_data(self, visual_objects: Dict[str, Any], physics_objects: Dict[str, Any]):
        """Merge physics data with visual data"""
        
        for obj_id in visual_objects.keys():
            if obj_id in physics_objects:
                physics_data = physics_objects[obj_id]
                
                # Update with accurate physics data
                visual_objects[obj_id].update({
                    'mass': physics_data.get('mass', 1.0),
                    'material': physics_data.get('material', 'unknown'),
                    'type': physics_data.get('type', 'object'),
                    'acceleration': physics_data.get('acceleration', [0, 0, 0])
                })
    
    def _extract_object_properties(self, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract properties from object data"""
        
        properties = {}
        
        # Extract basic properties
        if 'mass' in obj_data:
            properties['mass'] = obj_data['mass']
        
        if 'color' in obj_data:
            properties['color'] = obj_data['color']
        
        if 'material' in obj_data:
            properties['material'] = obj_data['material']
        
        if 'type' in obj_data:
            properties['object_type'] = obj_data['type']
        
        # Calculate derived properties
        if 'velocity' in obj_data:
            velocity = obj_data['velocity']
            if isinstance(velocity, (list, tuple, np.ndarray)):
                speed = np.linalg.norm(velocity)
                properties['speed'] = speed
                properties['is_moving'] = speed > self.velocity_threshold
        
        return properties
    
    def _update_object_tracker(self, obj_id: str, obj_data: Dict[str, Any]):
        """Update object tracking history"""
        
        current_time = time.time()
        
        if obj_id not in self.object_tracker:
            self.object_tracker[obj_id] = {
                'first_seen': current_time,
                'position_history': [],
                'velocity_history': [],
                'property_changes': []
            }
        
        tracker = self.object_tracker[obj_id]
        
        # Record position history
        tracker['position_history'].append({
            'position': obj_data['position'],
            'timestamp': current_time
        })
        
        # Record velocity history
        tracker['velocity_history'].append({
            'velocity': obj_data['velocity'],
            'timestamp': current_time
        })
        
        # Keep history manageable
        if len(tracker['position_history']) > 50:
            tracker['position_history'] = tracker['position_history'][-25:]
        
        if len(tracker['velocity_history']) > 50:
            tracker['velocity_history'] = tracker['velocity_history'][-25:]
    
    def _detect_motion_events(self, objects: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect motion events in objects"""
        
        motion_events = []
        current_time = time.time()
        
        for obj_id, obj_data in objects.items():
            if obj_id in self.object_tracker:
                tracker = self.object_tracker[obj_id]
                
                # Check for motion start
                if len(tracker['velocity_history']) >= 2:
                    prev_velocity = tracker['velocity_history'][-2]['velocity']
                    curr_velocity = tracker['velocity_history'][-1]['velocity']
                    
                    prev_speed = np.linalg.norm(prev_velocity)
                    curr_speed = np.linalg.norm(curr_velocity)
                    
                    # Motion started
                    if prev_speed <= self.velocity_threshold and curr_speed > self.velocity_threshold:
                        motion_events.append({
                            'type': 'motion_start',
                            'object_id': obj_id,
                            'velocity': curr_velocity,
                            'timestamp': current_time
                        })
                    
                    # Motion stopped
                    elif prev_speed > self.velocity_threshold and curr_speed <= self.velocity_threshold:
                        motion_events.append({
                            'type': 'motion_stop',
                            'object_id': obj_id,
                            'final_position': obj_data['position'],
                            'timestamp': current_time
                        })
        
        return motion_events
    
    def _detect_collision_events(self, objects: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect collision events between objects"""
        
        collision_events = []
        current_time = time.time()
        
        # Simple collision detection based on position proximity
        obj_list = list(objects.items())
        
        for i in range(len(obj_list)):
            for j in range(i + 1, len(obj_list)):
                obj1_id, obj1_data = obj_list[i]
                obj2_id, obj2_data = obj_list[j]
                
                if self._objects_colliding(obj1_data, obj2_data):
                    collision_events.append({
                        'type': 'collision',
                        'object1': obj1_id,
                        'object2': obj2_id,
                        'position': obj1_data['position'],
                        'timestamp': current_time
                    })
        
        return collision_events
    
    def _objects_colliding(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
        """Check if two objects are colliding"""
        
        pos1 = np.array(obj1['position'])
        pos2 = np.array(obj2['position'])
        
        distance = np.linalg.norm(pos2 - pos1)
        
        # Simple collision threshold
        collision_threshold = 1.0
        
        return distance < collision_threshold
    
    def get_object_motion_summary(self, obj_id: str) -> Dict[str, Any]:
        """Get motion summary for specific object"""
        
        if obj_id not in self.object_tracker:
            return {'error': 'Object not tracked'}
        
        tracker = self.object_tracker[obj_id]
        
        # Calculate motion statistics
        if len(tracker['velocity_history']) > 0:
            velocities = [np.linalg.norm(v['velocity']) for v in tracker['velocity_history']]
            
            return {
                'object_id': obj_id,
                'total_observations': len(tracker['position_history']),
                'average_speed': np.mean(velocities),
                'max_speed': np.max(velocities),
                'currently_moving': velocities[-1] > self.velocity_threshold if velocities else False,
                'tracking_duration': time.time() - tracker['first_seen']
            }
        
        return {'object_id': obj_id, 'no_motion_data': True}
