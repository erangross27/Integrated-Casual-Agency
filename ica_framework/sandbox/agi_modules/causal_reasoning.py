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
        self.temporal_patterns = []
        
        # Causal reasoning parameters
        self.temporal_window = 5.0  # seconds
        self.confidence_threshold = 0.7
    
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
        
        # Extract collision events
        sensory = observation.get('sensory_input', {})
        if 'visual' in sensory:
            objects = sensory.get('visual', {})
            for obj_id, obj_data in objects.items():
                if obj_data.get('collision_detected', False):
                    events.append({
                        'type': 'collision',
                        'object': obj_id,
                        'timestamp': time.time(),
                        'data': obj_data
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
        
        # Temporal ordering (cause before effect)
        if event1['timestamp'] >= event2['timestamp']:
            return False
        
        # Time window constraint
        if event2['timestamp'] - event1['timestamp'] > self.temporal_window:
            return False
        
        # Event type compatibility
        causal_patterns = [
            ('force_applied', 'collision'),
            ('collision', 'velocity_change'),
            ('velocity_change', 'position_change')
        ]
        
        for cause_type, effect_type in causal_patterns:
            if event1['type'] == cause_type and event2['type'] == effect_type:
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
