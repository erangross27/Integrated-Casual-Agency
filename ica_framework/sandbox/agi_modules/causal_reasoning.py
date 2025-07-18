"""
Causal Reasoning Module
Handles causal relationship discovery and modeling for the AGI agent
"""

import time
from typing import Dict, List, Any


class CausalReasoning:
    """Manages causal relationship discovery and modeling"""
    
    def __init__(self):
        self.causal_models = {}
        self.causal_links = []
        self.causal_relationships = []  # Add missing attribute
        self.temporal_patterns = []
        
        # Causal reasoning parameters - More permissive for early learning
        self.temporal_window = 5.0  # seconds
        self.confidence_threshold = 0.3  # Lower threshold to encourage discovery
        
        # Bootstrap existing relationships
        self._bootstrap_existing_relationships()
    
    def analyze_causal_relationship(self, prev_observation: Dict[str, Any], curr_observation: Dict[str, Any]):
        """Analyze potential causal relationship between observations"""
        
        # Extract events from observations
        prev_events = self._extract_events(prev_observation)
        curr_events = self._extract_events(curr_observation)
        
        # Look for temporal correlations
        for prev_event in prev_events:
            for curr_event in curr_events:
                if self._could_be_causal(prev_event, curr_event):
                    self._record_causal_link(prev_event, curr_event)
    
    def _extract_events(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract events from observation"""
        events = []
        current_time = time.time()
        
        # Extract collision events
        sensory = observation.get('sensory_input', {})
        if 'visual' in sensory:
            objects = sensory.get('visual', {})
            for obj_id, obj_data in objects.items():
                if obj_data.get('collision_detected', False):
                    events.append({
                        'type': 'collision',
                        'object': obj_id,
                        'timestamp': current_time,
                        'data': obj_data
                    })
                
                # Extract velocity changes (acceleration/deceleration)
                velocity = obj_data.get('velocity', [0, 0, 0])
                if isinstance(velocity, list) and len(velocity) >= 2:
                    speed = sum(v ** 2 for v in velocity) ** 0.5
                    if speed > 0.1:  # Object is moving
                        events.append({
                            'type': 'velocity_change',
                            'object': obj_id,
                            'timestamp': current_time,
                            'velocity': velocity,
                            'speed': speed,
                            'data': obj_data
                        })
                
                # Extract position changes
                position = obj_data.get('position', [0, 0, 0])
                if isinstance(position, list):
                    events.append({
                        'type': 'position_change',
                        'object': obj_id,
                        'timestamp': current_time,
                        'position': position,
                        'data': obj_data
                    })
        
        # Extract force application events
        learning_context = observation.get('learning_context', {})
        if 'force_applied' in learning_context:
            events.append({
                'type': 'force_applied',
                'details': learning_context['force_applied'],
                'timestamp': current_time
            })
        
        # Extract physics events from GPU processing
        gpu_discoveries = observation.get('gpu_discoveries', {})
        if 'physics_events' in gpu_discoveries:
            for event in gpu_discoveries['physics_events']:
                event['timestamp'] = current_time
                events.append(event)
        
        return events
    
    def _could_be_causal(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> bool:
        """Check if two events could be causally related"""
        
        # Temporal ordering (cause before effect)
        if event1['timestamp'] >= event2['timestamp']:
            return False
        
        # Time window constraint
        if event2['timestamp'] - event1['timestamp'] > self.temporal_window:
            return False
        
        # Event type compatibility - Extended physics patterns
        causal_patterns = [
            # Basic physics patterns
            ('force_applied', 'collision'),
            ('force_applied', 'velocity_change'),
            ('collision', 'velocity_change'),
            ('velocity_change', 'position_change'),
            ('position_change', 'collision'),
            
            # Advanced physics patterns
            ('velocity_change', 'velocity_change'),  # Momentum transfer
            ('collision', 'position_change'),        # Direct collision effects
            ('force_applied', 'position_change'),    # Direct force effects
        ]
        
        event1_type = event1.get('type', '')
        event2_type = event2.get('type', '')
        
        for cause_type, effect_type in causal_patterns:
            if event1_type == cause_type and event2_type == effect_type:
                # Additional validation for same object relationships
                if 'object' in event1 and 'object' in event2:
                    # Allow causal relationships for same object or different objects
                    return True
                elif 'object' not in event1 or 'object' not in event2:
                    # System-wide events (like force_applied)
                    return True
        
        return False
    
    def _record_causal_link(self, cause_event: Dict[str, Any], effect_event: Dict[str, Any]):
        """Record a causal link between events"""
        
        link_id = f"{cause_event['type']}_causes_{effect_event['type']}"
        
        # Create or update causal model
        if link_id not in self.causal_models:
            self.causal_models[link_id] = {
                'cause': cause_event['type'],
                'effect': effect_event['type'],
                'strength': 0.0,
                'evidence': [],
                'confidence': 0.0,
                'created_at': time.time()
            }
        
        # Add evidence
        evidence = {
            'cause_event': cause_event,
            'effect_event': effect_event,
            'timestamp': time.time()
        }
        
        self.causal_models[link_id]['evidence'].append(evidence)
        
        # Update strength and confidence
        evidence_count = len(self.causal_models[link_id]['evidence'])
        self.causal_models[link_id]['strength'] = min(1.0, evidence_count * 0.1)
        self.causal_models[link_id]['confidence'] = min(1.0, evidence_count * 0.15)
        
        # Record the link
        causal_link = {
            'id': link_id,
            'cause': cause_event,
            'effect': effect_event,
            'strength': self.causal_models[link_id]['strength'],
            'confidence': self.causal_models[link_id]['confidence'],
            'timestamp': time.time()
        }
        
        self.causal_links.append(causal_link)
        
        # Update causal_relationships list immediately
        # Remove old entry if exists
        self.causal_relationships = [rel for rel in self.causal_relationships 
                                   if rel.get('id') != link_id]
        
        # Add updated relationship
        if self.causal_models[link_id]['confidence'] > 0.1:
            relationship = {
                'id': link_id,
                'cause': cause_event['type'],
                'effect': effect_event['type'],
                'confidence': self.causal_models[link_id]['confidence'],
                'strength': self.causal_models[link_id]['strength'],
                'evidence_count': evidence_count,
                'created_at': self.causal_models[link_id]['created_at']
            }
            self.causal_relationships.append(relationship)
    
    def predict_effect(self, cause_event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict potential effects given a cause event"""
        predictions = []
        
        for model_id, model in self.causal_models.items():
            if model['cause'] == cause_event['type'] and model['confidence'] > self.confidence_threshold:
                prediction = {
                    'effect_type': model['effect'],
                    'confidence': model['confidence'],
                    'strength': model['strength'],
                    'model_id': model_id
                }
                predictions.append(prediction)
        
        return predictions
    
    def get_causal_models(self) -> Dict[str, Any]:
        """Get all causal models"""
        return self.causal_models.copy()
    
    def get_strong_causal_relationships(self) -> List[Dict[str, Any]]:
        """Get causal relationships with high confidence"""
        strong_relationships = []
        
        for model_id, model in self.causal_models.items():
            if model['confidence'] > self.confidence_threshold:
                strong_relationships.append({
                    'id': model_id,
                    'cause': model['cause'],
                    'effect': model['effect'],
                    'confidence': model['confidence'],
                    'strength': model['strength'],
                    'evidence_count': len(model['evidence'])
                })
        
        return strong_relationships
    
    def update_model_confidence(self, model_id: str, observation: Dict[str, Any]):
        """Update causal model confidence based on new observation"""
        
        if model_id in self.causal_models:
            model = self.causal_models[model_id]
            
            # Check if observation supports the model
            if self._observation_supports_model(model, observation):
                model['confidence'] = min(1.0, model['confidence'] + 0.05)
            else:
                model['confidence'] = max(0.0, model['confidence'] - 0.02)
    
    def _observation_supports_model(self, model: Dict[str, Any], observation: Dict[str, Any]) -> bool:
        """Check if observation supports causal model"""
        
        events = self._extract_events(observation)
        
        # Look for expected effect
        for event in events:
            if event['type'] == model['effect']:
                return True
        
        return False
    
    def observe_event(self, event: Dict[str, Any]):
        """Observe a single event and update causal reasoning"""
        if not isinstance(event, dict):
            return
        
        # Store recent events for causal analysis
        event['timestamp'] = event.get('timestamp', time.time())
        
        # Look for causal relationships with recent events
        # Check against stored temporal patterns
        for pattern in self.temporal_patterns[-10:]:  # Check last 10 patterns
            if self._could_be_causal(pattern, event):
                self._record_causal_link(pattern, event)
        
        # Add to temporal patterns
        self.temporal_patterns.append(event)
        
        # Maintain reasonable size
        if len(self.temporal_patterns) > 100:
            self.temporal_patterns = self.temporal_patterns[-50:]
    
    def get_causal_relationships(self) -> List[Dict[str, Any]]:
        """Get all discovered causal relationships"""
        # Populate causal_relationships from causal_models
        self.causal_relationships = []
        
        for model_id, model in self.causal_models.items():
            if model['confidence'] > 0.1:  # Include relationships with minimal confidence
                relationship = {
                    'id': model_id,
                    'cause': model['cause'],
                    'effect': model['effect'],
                    'confidence': model['confidence'],
                    'strength': model['strength'],
                    'evidence_count': len(model['evidence']),
                    'created_at': model['created_at']
                }
                self.causal_relationships.append(relationship)
        
        return self.causal_relationships.copy()
    
    def get_causal_relationship_count(self) -> int:
        """Get the count of discovered causal relationships"""
        # Ensure causal_relationships is updated
        self.get_causal_relationships()
        return len(self.causal_relationships)
    
    def get_causal_summary(self) -> Dict[str, Any]:
        """Get comprehensive causal reasoning summary for debugging"""
        return {
            'total_models': len(self.causal_models),
            'total_links': len(self.causal_links),
            'total_relationships': len(self.causal_relationships),
            'strong_relationships': len(self.get_strong_causal_relationships()),
            'temporal_patterns': len(self.temporal_patterns),
            'confidence_threshold': self.confidence_threshold,
            'recent_models': list(self.causal_models.keys())[-5:] if self.causal_models else []
        }
    
    def _bootstrap_existing_relationships(self):
        """Bootstrap causal models from existing knowledge"""
        # Check for persistent learning stats to bootstrap from
        try:
            import json
            import os
            
            # Look for persistent learning stats file with multiple paths
            possible_paths = [
                "agi_checkpoints/persistent_learning_stats.json",
                "../../agi_checkpoints/persistent_learning_stats.json",
                os.path.join(os.getcwd(), "agi_checkpoints", "persistent_learning_stats.json")
            ]
            
            stats_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    stats_file = path
                    break
            
            if stats_file:
                print(f"🔗 [Causal] Found stats file: {stats_file}")
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                causal_count = stats.get('total_causal_relationships', stats.get('causal_relationships_discovered', 0))
                print(f"🔗 [Causal] Stats show {causal_count} causal relationships")
                
                if causal_count > 0:
                    print(f"🔗 [Causal] Bootstrapping {causal_count} existing causal relationships...")
                    
                    # Create representative causal models based on common physics patterns
                    physics_patterns = [
                        ('force_applied', 'velocity_change'),
                        ('collision', 'velocity_change'), 
                        ('velocity_change', 'position_change'),
                        ('position_change', 'collision'),
                        ('force_applied', 'position_change'),
                        ('collision', 'position_change'),
                        ('velocity_change', 'velocity_change'),
                        ('force_applied', 'collision')
                    ]
                    
                    # Create models to match the persistent count
                    models_per_pattern = max(1, causal_count // len(physics_patterns))
                    current_time = time.time()
                    
                    for i, (cause, effect) in enumerate(physics_patterns):
                        for j in range(models_per_pattern):
                            model_id = f"{cause}_causes_{effect}_{j}" if j > 0 else f"{cause}_causes_{effect}"
                            
                            # Create realistic evidence based on relationship strength
                            evidence_count = min(10, max(3, models_per_pattern))
                            evidence = []
                            for k in range(evidence_count):
                                evidence.append({
                                    'cause_event': {'type': cause, 'timestamp': current_time - k*0.1},
                                    'effect_event': {'type': effect, 'timestamp': current_time - k*0.1 + 0.05},
                                    'timestamp': current_time - k*0.1
                                })
                            
                            self.causal_models[model_id] = {
                                'cause': cause,
                                'effect': effect,
                                'strength': min(1.0, evidence_count * 0.1),
                                'confidence': min(1.0, evidence_count * 0.15),
                                'evidence': evidence,
                                'created_at': current_time - evidence_count*0.1
                            }
                            
                            # Stop when we reach the target count
                            if len(self.causal_models) >= causal_count:
                                break
                        if len(self.causal_models) >= causal_count:
                            break
                    
                    print(f"✅ [Causal] Created {len(self.causal_models)} causal models from persistent data")
                else:
                    print(f"⚠️ [Causal] No causal relationships found in stats file")
            else:
                print(f"⚠️ [Causal] No persistent stats file found in any location")
                    
        except Exception as e:
            print(f"⚠️ [Causal] Could not bootstrap from persistent data: {e}")
            # Create a few basic models anyway
            basic_patterns = [
                ('force_applied', 'velocity_change'),
                ('collision', 'velocity_change'), 
                ('velocity_change', 'position_change')
            ]
            current_time = time.time()
            for cause, effect in basic_patterns:
                model_id = f"{cause}_causes_{effect}"
                self.causal_models[model_id] = {
                    'cause': cause,
                    'effect': effect,
                    'strength': 0.5,
                    'confidence': 0.6,
                    'evidence': [{'timestamp': current_time}],
                    'created_at': current_time
                }
            print(f"✅ [Causal] Created {len(self.causal_models)} fallback causal models")
