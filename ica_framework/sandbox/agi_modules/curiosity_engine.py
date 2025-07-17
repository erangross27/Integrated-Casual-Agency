"""
Curiosity Engine Module
Drives curiosity-based learning and exploration
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class CuriosityEngine:
    """Generates curiosity-driven questions and exploration targets"""
    
    def __init__(self):
        self.curiosity_targets = []
        self.generated_questions = []
        self.exploration_interests = {}
        self.knowledge_gaps = {}
        
        # Curiosity parameters
        self.max_curiosity_targets = 10
        self.novelty_threshold = 0.6
        self.question_generation_rate = 0.3
        self.curiosity_decay_rate = 0.98
        self.gap_importance_threshold = 0.4
    
    def assess_curiosity(self, observations: List[Dict[str, Any]], 
                        current_knowledge: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess what aspects of observations generate curiosity"""
        
        curiosity_results = {
            'curiosity_targets': [],
            'generated_questions': [],
            'knowledge_gaps': [],
            'exploration_suggestions': []
        }
        
        for observation in observations:
            # Check for novel or unexpected aspects
            novelty_score = self._assess_novelty(observation, current_knowledge)
            
            if novelty_score > self.novelty_threshold:
                curiosity_target = self._create_curiosity_target(observation, novelty_score)
                curiosity_results['curiosity_targets'].append(curiosity_target)
                
                # Generate questions about novel observations
                questions = self._generate_questions_about(observation)
                curiosity_results['generated_questions'].extend(questions)
        
        # Identify knowledge gaps
        gaps = self._identify_knowledge_gaps(observations, current_knowledge)
        curiosity_results['knowledge_gaps'] = gaps
        
        # Generate exploration suggestions
        suggestions = self._generate_exploration_suggestions(curiosity_results['curiosity_targets'])
        curiosity_results['exploration_suggestions'] = suggestions
        
        # Update internal state
        self._update_curiosity_state(curiosity_results)
        
        return curiosity_results
    
    def _assess_novelty(self, observation: Dict[str, Any], 
                       current_knowledge: Dict[str, Any] = None) -> float:
        """Assess how novel or surprising an observation is"""
        
        novelty_score = 0.0
        
        # Check if we've seen this type of object/event before
        obs_type = observation.get('type', 'unknown')
        if obs_type not in self.exploration_interests:
            novelty_score += 0.8  # New type is highly novel
        else:
            # Familiar type, but check for unusual properties
            familiarity = self.exploration_interests[obs_type]['familiarity']
            novelty_score += max(0.0, 0.5 - familiarity)
        
        # Check for unexpected properties
        expected_properties = self._get_expected_properties(obs_type, current_knowledge)
        actual_properties = observation.get('properties', {})
        
        property_surprises = 0
        for prop, value in actual_properties.items():
            expected_value = expected_properties.get(prop, None)
            if expected_value is None:
                property_surprises += 1  # Unexpected property
            elif self._is_value_surprising(value, expected_value):
                property_surprises += 0.5  # Unexpected value
        
        novelty_score += min(0.5, property_surprises * 0.1)
        
        # Check for unusual behavior/motion
        velocity = observation.get('velocity', [0, 0, 0])
        motion_magnitude = np.linalg.norm(velocity)
        if motion_magnitude > 5.0:  # Fast motion is interesting
            novelty_score += 0.3
        
        # Check for interactions with other objects
        interactions = observation.get('interactions', [])
        if interactions:
            novelty_score += min(0.3, len(interactions) * 0.1)
        
        # Environmental context novelty
        position = observation.get('position', [0, 0, 0])
        if self._is_unusual_location(position, obs_type):
            novelty_score += 0.2
        
        return min(1.0, novelty_score)
    
    def _create_curiosity_target(self, observation: Dict[str, Any], novelty_score: float) -> Dict[str, Any]:
        """Create a curiosity target from a novel observation"""
        
        return {
            'id': f"curiosity_{int(time.time())}_{random.randint(1000, 9999)}",
            'object_id': observation.get('id', 'unknown'),
            'type': observation.get('type', 'unknown'),
            'position': observation.get('position', [0, 0, 0]),
            'novelty_score': novelty_score,
            'curiosity_level': novelty_score,
            'timestamp': time.time(),
            'investigation_priority': min(1.0, novelty_score * 1.2),
            'questions_generated': 0,
            'investigation_time': 0.0,
            'reason': self._generate_curiosity_reason(observation, novelty_score)
        }
    
    def _generate_curiosity_reason(self, observation: Dict[str, Any], novelty_score: float) -> str:
        """Generate a reason why this observation is interesting"""
        
        reasons = []
        
        obs_type = observation.get('type', 'unknown')
        
        if novelty_score > 0.8:
            reasons.append(f"Never seen a {obs_type} before")
        elif novelty_score > 0.6:
            reasons.append(f"Unusual properties in {obs_type}")
        
        velocity = observation.get('velocity', [0, 0, 0])
        if np.linalg.norm(velocity) > 5.0:
            reasons.append("Moving very fast")
        
        interactions = observation.get('interactions', [])
        if interactions:
            reasons.append(f"Interacting with {len(interactions)} other objects")
        
        properties = observation.get('properties', {})
        unusual_props = []
        for prop, value in properties.items():
            if isinstance(value, (int, float)) and value > 10:
                unusual_props.append(prop)
        
        if unusual_props:
            reasons.append(f"Unusual {', '.join(unusual_props[:2])}")
        
        return "; ".join(reasons) if reasons else "Generally interesting object"
    
    def _generate_questions_about(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate questions about an observation"""
        
        questions = []
        obs_type = observation.get('type', 'unknown')
        
        # Question templates based on observation type and properties
        question_templates = [
            f"What causes {obs_type} to behave this way?",
            f"How does {obs_type} interact with its environment?",
            f"What properties determine {obs_type}'s behavior?",
            f"Can {obs_type}'s behavior be predicted?",
            f"What happens if {obs_type} encounters different conditions?"
        ]
        
        # Generate specific questions based on observation properties
        velocity = observation.get('velocity', [0, 0, 0])
        if np.linalg.norm(velocity) > 0.1:
            question_templates.append(f"What determines the speed of {obs_type}?")
            question_templates.append(f"Will {obs_type} continue moving or stop?")
        
        position = observation.get('position', [0, 0, 0])
        if position[1] > 5.0:  # High altitude
            question_templates.append(f"Why is {obs_type} so high up?")
            question_templates.append(f"Will {obs_type} fall down?")
        
        interactions = observation.get('interactions', [])
        if interactions:
            question_templates.append(f"How does {obs_type} affect other objects?")
            question_templates.append(f"What determines {obs_type}'s interaction patterns?")
        
        # Select a subset of questions
        selected_questions = random.sample(
            question_templates, 
            min(3, len(question_templates))
        )
        
        for question_text in selected_questions:
            question = {
                'id': f"question_{int(time.time())}_{random.randint(100, 999)}",
                'text': question_text,
                'type': 'curiosity_driven',
                'target_object': observation.get('id', 'unknown'),
                'timestamp': time.time(),
                'investigation_methods': self._suggest_investigation_methods(question_text),
                'priority': random.uniform(0.4, 0.8)
            }
            questions.append(question)
        
        return questions
    
    def _suggest_investigation_methods(self, question: str) -> List[str]:
        """Suggest methods to investigate a question"""
        
        methods = []
        
        if "behavior" in question.lower():
            methods.extend(["observe_over_time", "change_environment", "measure_patterns"])
        
        if "interact" in question.lower():
            methods.extend(["introduce_stimulus", "remove_obstacles", "test_responses"])
        
        if "properties" in question.lower():
            methods.extend(["measure_attributes", "compare_with_similar", "test_boundaries"])
        
        if "predict" in question.lower():
            methods.extend(["collect_data", "build_model", "test_predictions"])
        
        if "speed" in question.lower() or "move" in question.lower():
            methods.extend(["track_motion", "measure_forces", "analyze_trajectory"])
        
        # Default methods if no specific matches
        if not methods:
            methods = ["observe_closely", "test_hypothesis", "gather_data"]
        
        return list(set(methods))  # Remove duplicates
    
    def _identify_knowledge_gaps(self, observations: List[Dict[str, Any]], 
                                current_knowledge: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Identify gaps in current knowledge"""
        
        gaps = []
        
        if current_knowledge is None:
            current_knowledge = {}
        
        # Look for phenomena we can observe but can't explain
        for observation in observations:
            obs_type = observation.get('type', 'unknown')
            
            # Check if we have explanatory knowledge about this type
            knowledge_about_type = current_knowledge.get(obs_type, {})
            
            if not knowledge_about_type:
                gap = {
                    'type': 'unknown_object_type',
                    'description': f"No knowledge about {obs_type}",
                    'object_type': obs_type,
                    'importance': 0.8,
                    'investigation_priority': 0.7,
                    'timestamp': time.time()
                }
                gaps.append(gap)
            
            # Check for unexplained behaviors
            velocity = observation.get('velocity', [0, 0, 0])
            if np.linalg.norm(velocity) > 0.1:
                motion_knowledge = knowledge_about_type.get('motion_patterns', {})
                if not motion_knowledge:
                    gap = {
                        'type': 'unexplained_motion',
                        'description': f"Don't understand {obs_type} motion patterns",
                        'object_type': obs_type,
                        'importance': 0.6,
                        'investigation_priority': 0.5,
                        'timestamp': time.time()
                    }
                    gaps.append(gap)
            
            # Check for unexplained interactions
            interactions = observation.get('interactions', [])
            if interactions:
                interaction_knowledge = knowledge_about_type.get('interaction_rules', {})
                if not interaction_knowledge:
                    gap = {
                        'type': 'unexplained_interactions',
                        'description': f"Don't understand how {obs_type} interacts with others",
                        'object_type': obs_type,
                        'importance': 0.7,
                        'investigation_priority': 0.6,
                        'timestamp': time.time()
                    }
                    gaps.append(gap)
        
        return gaps
    
    def _generate_exploration_suggestions(self, curiosity_targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate suggestions for curiosity-driven exploration"""
        
        suggestions = []
        
        for target in curiosity_targets:
            # Suggest getting closer to interesting objects
            suggestions.append({
                'type': 'approach_target',
                'target_id': target['object_id'],
                'description': f"Move closer to investigate {target['type']}",
                'position': target['position'],
                'priority': target['investigation_priority'],
                'estimated_duration': 2.0
            })
            
            # Suggest observing for extended time
            if target['novelty_score'] > 0.7:
                suggestions.append({
                    'type': 'extended_observation',
                    'target_id': target['object_id'],
                    'description': f"Observe {target['type']} for extended period",
                    'position': target['position'],
                    'priority': target['investigation_priority'] * 0.8,
                    'estimated_duration': 5.0
                })
            
            # Suggest interaction experiments
            if target['curiosity_level'] > 0.8:
                suggestions.append({
                    'type': 'interaction_experiment',
                    'target_id': target['object_id'],
                    'description': f"Test interactions with {target['type']}",
                    'position': target['position'],
                    'priority': target['investigation_priority'] * 0.9,
                    'estimated_duration': 3.0
                })
        
        # Sort suggestions by priority
        suggestions.sort(key=lambda x: x['priority'], reverse=True)
        
        return suggestions[:self.max_curiosity_targets]
    
    def _update_curiosity_state(self, curiosity_results: Dict[str, Any]):
        """Update internal curiosity state"""
        
        # Add new curiosity targets
        new_targets = curiosity_results['curiosity_targets']
        self.curiosity_targets.extend(new_targets)
        
        # Limit number of active targets
        if len(self.curiosity_targets) > self.max_curiosity_targets:
            # Keep highest priority targets
            self.curiosity_targets.sort(key=lambda x: x['investigation_priority'], reverse=True)
            self.curiosity_targets = self.curiosity_targets[:self.max_curiosity_targets]
        
        # Add new questions
        new_questions = curiosity_results['generated_questions']
        self.generated_questions.extend(new_questions)
        
        # Update knowledge gaps
        new_gaps = curiosity_results['knowledge_gaps']
        for gap in new_gaps:
            gap_key = f"{gap['type']}_{gap['object_type']}"
            self.knowledge_gaps[gap_key] = gap
        
        # Decay curiosity over time
        current_time = time.time()
        for target in self.curiosity_targets:
            age = current_time - target['timestamp']
            decay_factor = self.curiosity_decay_rate ** (age / 60.0)  # Decay per minute
            target['curiosity_level'] *= decay_factor
        
        # Remove very low curiosity targets
        self.curiosity_targets = [t for t in self.curiosity_targets if t['curiosity_level'] > 0.1]
    
    def _get_expected_properties(self, obj_type: str, knowledge: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get expected properties for an object type"""
        
        if knowledge is None or obj_type not in knowledge:
            return {}
        
        return knowledge[obj_type].get('expected_properties', {})
    
    def _is_value_surprising(self, actual_value: Any, expected_value: Any) -> bool:
        """Check if an actual value is surprising compared to expected"""
        
        if isinstance(actual_value, (int, float)) and isinstance(expected_value, (int, float)):
            # Numeric comparison
            if expected_value == 0:
                return abs(actual_value) > 0.1
            else:
                relative_diff = abs(actual_value - expected_value) / abs(expected_value)
                return relative_diff > 0.5
        
        # Non-numeric comparison
        return actual_value != expected_value
    
    def _is_unusual_location(self, position: List[float], obj_type: str) -> bool:
        """Check if an object is in an unusual location for its type"""
        
        # Simple heuristics for unusual locations
        x, y, z = position
        
        # Very high or very low positions are unusual
        if y > 20.0 or y < -5.0:
            return True
        
        # Very far from origin is unusual
        distance_from_origin = np.linalg.norm(position)
        if distance_from_origin > 50.0:
            return True
        
        return False
    
    def get_curiosity_summary(self) -> Dict[str, Any]:
        """Get summary of curiosity system state"""
        
        return {
            'active_curiosity_targets': len(self.curiosity_targets),
            'total_questions_generated': len(self.generated_questions),
            'knowledge_gaps_identified': len(self.knowledge_gaps),
            'average_curiosity_level': np.mean([t['curiosity_level'] for t in self.curiosity_targets]) 
                                     if self.curiosity_targets else 0.0,
            'highest_priority_target': max(self.curiosity_targets, 
                                         key=lambda x: x['investigation_priority'])['id'] 
                                     if self.curiosity_targets else None
        }
    
    def get_investigation_recommendations(self) -> List[Dict[str, Any]]:
        """Get current recommendations for curiosity-driven investigation"""
        
        if not self.curiosity_targets:
            return []
        
        # Sort targets by investigation priority
        sorted_targets = sorted(self.curiosity_targets, 
                              key=lambda x: x['investigation_priority'], 
                              reverse=True)
        
        recommendations = []
        for target in sorted_targets[:5]:  # Top 5 recommendations
            recommendation = {
                'action': 'investigate',
                'target_id': target['object_id'],
                'target_type': target['type'],
                'position': target['position'],
                'priority': target['investigation_priority'],
                'curiosity_level': target['curiosity_level'],
                'reason': target['reason'],
                'methods': ['observe_closely', 'test_interactions', 'measure_properties']
            }
            recommendations.append(recommendation)
        
        return recommendations
