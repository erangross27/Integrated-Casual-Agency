"""
Hypothesis Manager Module
Manages hypothesis formation, testing, and confirmation for the AGI agent
"""

import time
from typing import Dict, List, Any, Optional


class HypothesisManager:
    """Manages AGI hypothesis formation and testing"""
    
    def __init__(self):
        self.active_hypotheses = []
        self.tested_hypotheses = []
        self.confirmed_hypotheses = []
        self.rejected_hypotheses = []
        
        # Hypothesis generation parameters
        self.confidence_threshold = 0.6
        self.evidence_threshold = 3
    
    def generate_hypothesis(self, observation1: Dict[str, Any], observation2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate hypothesis from two observations"""
        
        # Look for cause-effect relationships
        if self._has_significant_change(observation1, observation2):
            hypothesis = {
                'id': f"hyp_{len(self.active_hypotheses) + 1}_{int(time.time())}",
                'description': self._generate_description(observation1, observation2),
                'type': 'causal_hypothesis',
                'evidence': [observation1, observation2],
                'confidence': 0.5,
                'testable': True,
                'tested': False,
                'timestamp': time.time(),
                'source': 'observation_comparison'
            }
            
            self.active_hypotheses.append(hypothesis)
            return hypothesis
        
        return None
    
    def test_hypothesis(self, hypothesis: Dict[str, Any], observation: Dict[str, Any]) -> Optional[bool]:
        """Test hypothesis against new observation"""
        
        if hypothesis['tested']:
            return hypothesis.get('confirmed', False)
        
        # Simple pattern matching test
        if self._hypothesis_matches_observation(hypothesis, observation):
            hypothesis['confidence'] += 0.1
            hypothesis['evidence'].append(observation)
        else:
            hypothesis['confidence'] -= 0.05
        
        # Check if we have enough evidence to make a decision
        if len(hypothesis['evidence']) >= self.evidence_threshold:
            if hypothesis['confidence'] >= self.confidence_threshold:
                return self._confirm_hypothesis(hypothesis)
            else:
                return self._reject_hypothesis(hypothesis)
        
        return None
    
    def _has_significant_change(self, obs1: Dict[str, Any], obs2: Dict[str, Any]) -> bool:
        """Check if observations show significant change"""
        
        # Check for object changes
        sensory1 = obs1.get('sensory_input', {})
        sensory2 = obs2.get('sensory_input', {})
        
        if 'visual' in sensory1 and 'visual' in sensory2:
            objects1 = sensory1.get('visual', {})
            objects2 = sensory2.get('visual', {})
            
            # Look for velocity or position changes
            for obj_id in objects1.keys():
                if obj_id in objects2:
                    obj1 = objects1[obj_id]
                    obj2 = objects2[obj_id]
                    
                    if self._object_changed_significantly(obj1, obj2):
                        return True
        
        return False
    
    def _object_changed_significantly(self, obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
        """Check if object changed significantly"""
        
        # Check position change
        if 'position' in obj1 and 'position' in obj2:
            pos1 = obj1['position']
            pos2 = obj2['position']
            
            if isinstance(pos1, list) and isinstance(pos2, list) and len(pos1) == len(pos2):
                distance = sum((p2 - p1) ** 2 for p1, p2 in zip(pos1, pos2)) ** 0.5
                if distance > 0.5:
                    return True
        
        # Check velocity change
        if 'velocity' in obj1 and 'velocity' in obj2:
            vel1 = obj1['velocity']
            vel2 = obj2['velocity']
            
            if isinstance(vel1, list) and isinstance(vel2, list) and len(vel1) == len(vel2):
                vel_change = sum((v2 - v1) ** 2 for v1, v2 in zip(vel1, vel2)) ** 0.5
                if vel_change > 1.0:
                    return True
        
        return False
    
    def _generate_description(self, obs1: Dict[str, Any], obs2: Dict[str, Any]) -> str:
        """Generate hypothesis description"""
        return f"Observed change between timesteps suggests causal relationship"
    
    def _hypothesis_matches_observation(self, hypothesis: Dict[str, Any], observation: Dict[str, Any]) -> bool:
        """Check if hypothesis matches observation"""
        
        # Simple pattern matching - in real AGI this would be more sophisticated
        if 'causal' in hypothesis['description'].lower():
            return self._has_causal_pattern(observation)
        
        return False
    
    def _has_causal_pattern(self, observation: Dict[str, Any]) -> bool:
        """Check for causal patterns in observation"""
        
        # Look for evidence of cause-effect relationships
        sensory = observation.get('sensory_input', {})
        if 'visual' in sensory:
            objects = sensory.get('visual', {})
            
            # Check for moving objects (effect of forces)
            for obj in objects.values():
                if 'velocity' in obj:
                    velocity = obj['velocity']
                    if isinstance(velocity, list):
                        speed = sum(v ** 2 for v in velocity) ** 0.5
                        if speed > 0.1:
                            return True
        
        return False
    
    def _confirm_hypothesis(self, hypothesis: Dict[str, Any]) -> bool:
        """Confirm a hypothesis"""
        hypothesis['tested'] = True
        hypothesis['confirmed'] = True
        hypothesis['confirmation_time'] = time.time()
        
        self.confirmed_hypotheses.append(hypothesis)
        if hypothesis in self.active_hypotheses:
            self.active_hypotheses.remove(hypothesis)
        
        return True
    
    def _reject_hypothesis(self, hypothesis: Dict[str, Any]) -> bool:
        """Reject a hypothesis"""
        hypothesis['tested'] = True
        hypothesis['confirmed'] = False
        hypothesis['rejection_time'] = time.time()
        
        self.rejected_hypotheses.append(hypothesis)
        if hypothesis in self.active_hypotheses:
            self.active_hypotheses.remove(hypothesis)
        
        return False
    
    def get_active_hypotheses(self) -> List[Dict[str, Any]]:
        """Get currently active hypotheses"""
        return [h for h in self.active_hypotheses if not h['tested']]
    
    def get_confirmed_hypotheses(self) -> List[Dict[str, Any]]:
        """Get confirmed hypotheses"""
        return self.confirmed_hypotheses.copy()
    
    def get_hypothesis_summary(self) -> Dict[str, Any]:
        """Get comprehensive hypothesis summary"""
        stats = self.get_hypothesis_stats()
        return {
            'stats': stats,
            'active_hypotheses': list(self.active_hypotheses.keys()),
            'confirmed_count': stats['confirmed'],
            'confidence_scores': {hid: h.get('confidence', 0.0) 
                                for hid, h in self.active_hypotheses.items()}
        }
    
    def get_hypothesis_stats(self) -> Dict[str, int]:
        """Get hypothesis statistics"""
        return {
            'active': len(self.active_hypotheses),
            'confirmed': len(self.confirmed_hypotheses),
            'rejected': len(self.rejected_hypotheses),
            'total_tested': len(self.confirmed_hypotheses) + len(self.rejected_hypotheses)
        }
