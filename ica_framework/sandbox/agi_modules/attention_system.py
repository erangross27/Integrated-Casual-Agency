"""
Attention System Module
Manages attention and focus on relevant stimuli and tasks
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


class AttentionSystem:
    """Manages selective attention and focus mechanisms"""
    
    def __init__(self):
        self.attention_targets = {}
        self.attention_history = []
        self.focus_weights = {}
        self.distraction_filters = {}
        
        # Attention parameters
        self.max_attention_targets = 7  # Miller's magic number
        self.attention_decay_rate = 0.95
        self.novelty_boost = 2.0
        self.urgency_boost = 3.0
        self.relevance_threshold = 0.3
    
    def process_stimuli(self, stimuli: List[Dict[str, Any]], 
                       current_goals: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process incoming stimuli and determine what deserves attention"""
        
        # Calculate attention scores for all stimuli
        attention_scores = []
        
        for stimulus in stimuli:
            score = self._calculate_attention_score(stimulus, current_goals)
            
            attention_scores.append({
                'stimulus': stimulus,
                'attention_score': score,
                'timestamp': time.time()
            })
        
        # Sort by attention score
        attention_scores.sort(key=lambda x: x['attention_score'], reverse=True)
        
        # Select top attention targets
        selected_targets = attention_scores[:self.max_attention_targets]
        
        # Update attention system state
        self._update_attention_targets(selected_targets)
        
        # Apply focus and filtering
        focused_attention = self._apply_attention_focus(selected_targets)
        
        return {
            'primary_focus': focused_attention['primary'] if focused_attention['primary'] else None,
            'secondary_targets': focused_attention['secondary'],
            'background_awareness': focused_attention['background'],
            'ignored_stimuli': len(stimuli) - len(selected_targets),
            'attention_distribution': self._get_attention_distribution()
        }
    
    def _calculate_attention_score(self, stimulus: Dict[str, Any], 
                                 current_goals: List[Dict[str, Any]] = None) -> float:
        """Calculate attention score for a stimulus"""
        
        score = 0.0
        
        # Base saliency score
        saliency = stimulus.get('saliency', 0.5)
        score += saliency
        
        # Novelty bonus
        novelty = stimulus.get('novelty', 0.0)
        score += novelty * self.novelty_boost
        
        # Urgency bonus
        urgency = stimulus.get('urgency', 0.0)
        score += urgency * self.urgency_boost
        
        # Goal relevance
        if current_goals:
            goal_relevance = self._assess_goal_relevance(stimulus, current_goals)
            score += goal_relevance * 2.0
        
        # Motion attention (moving objects attract attention)
        velocity = stimulus.get('velocity', [0, 0, 0])
        motion_magnitude = np.linalg.norm(velocity)
        if motion_magnitude > 0.1:
            score += min(1.0, motion_magnitude * 0.5)
        
        # Size/proximity factor
        distance = stimulus.get('distance', 10.0)
        size = stimulus.get('size', 1.0)
        proximity_factor = size / max(1.0, distance)
        score += proximity_factor
        
        # Previous attention bias (things we've focused on before)
        stimulus_id = stimulus.get('id', 'unknown')
        if stimulus_id in self.attention_targets:
            previous_attention = self.attention_targets[stimulus_id]['cumulative_attention']
            score += previous_attention * 0.3  # Mild bias toward familiar objects
        
        # Surprise/expectation violation
        surprise = stimulus.get('surprise', 0.0)
        score += surprise * 1.5
        
        return max(0.0, score)
    
    def _assess_goal_relevance(self, stimulus: Dict[str, Any], 
                             current_goals: List[Dict[str, Any]]) -> float:
        """Assess how relevant a stimulus is to current goals"""
        
        max_relevance = 0.0
        
        stimulus_type = stimulus.get('type', 'unknown')
        stimulus_position = stimulus.get('position', [0, 0, 0])
        
        for goal in current_goals:
            relevance = 0.0
            
            # Type matching
            goal_target_type = goal.get('target_type', None)
            if goal_target_type and goal_target_type == stimulus_type:
                relevance += 0.8
            
            # Spatial relevance
            goal_position = goal.get('position', None)
            if goal_position:
                distance = np.linalg.norm(np.array(stimulus_position) - np.array(goal_position))
                spatial_relevance = 1.0 / (1.0 + distance * 0.1)
                relevance += spatial_relevance * 0.6
            
            # Keyword matching
            goal_keywords = goal.get('keywords', [])
            stimulus_description = stimulus.get('description', '')
            for keyword in goal_keywords:
                if keyword.lower() in stimulus_description.lower():
                    relevance += 0.4
                    break
            
            max_relevance = max(max_relevance, relevance)
        
        return max_relevance
    
    def _update_attention_targets(self, selected_targets: List[Dict[str, Any]]):
        """Update internal tracking of attention targets"""
        
        current_time = time.time()
        
        # Decay existing attention weights
        for target_id in self.attention_targets:
            self.attention_targets[target_id]['weight'] *= self.attention_decay_rate
            self.attention_targets[target_id]['last_update'] = current_time
        
        # Update attention for selected targets
        for target_data in selected_targets:
            stimulus = target_data['stimulus']
            score = target_data['attention_score']
            
            stimulus_id = stimulus.get('id', f'unknown_{hash(str(stimulus))}')
            
            if stimulus_id not in self.attention_targets:
                self.attention_targets[stimulus_id] = {
                    'weight': 0.0,
                    'cumulative_attention': 0.0,
                    'first_noticed': current_time,
                    'total_focus_time': 0.0,
                    'focus_episodes': 0
                }
            
            target = self.attention_targets[stimulus_id]
            target['weight'] = score
            target['cumulative_attention'] += score
            target['last_update'] = current_time
            
            # If this is high attention, count as focus episode
            if score > 2.0:
                target['focus_episodes'] += 1
                target['total_focus_time'] += 1.0  # Simplified time unit
        
        # Remove very old or weak attention targets
        self._cleanup_attention_targets()
    
    def _cleanup_attention_targets(self):
        """Remove old or irrelevant attention targets"""
        
        current_time = time.time()
        cleanup_threshold = 60.0  # 60 seconds
        weight_threshold = 0.1
        
        targets_to_remove = []
        
        for target_id, target_data in self.attention_targets.items():
            age = current_time - target_data['last_update']
            weight = target_data['weight']
            
            if age > cleanup_threshold or weight < weight_threshold:
                targets_to_remove.append(target_id)
        
        for target_id in targets_to_remove:
            del self.attention_targets[target_id]
    
    def _apply_attention_focus(self, selected_targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply focused attention mechanism to selected targets"""
        
        if not selected_targets:
            return {'primary': None, 'secondary': [], 'background': []}
        
        # Primary focus (highest scoring target)
        primary_target = selected_targets[0]['stimulus']
        
        # Secondary targets (next 2-3 highest scoring)
        secondary_targets = []
        for i in range(1, min(4, len(selected_targets))):
            if selected_targets[i]['attention_score'] > self.relevance_threshold:
                secondary_targets.append(selected_targets[i]['stimulus'])
        
        # Background awareness (remaining targets)
        background_targets = []
        for i in range(len(secondary_targets) + 1, len(selected_targets)):
            background_targets.append(selected_targets[i]['stimulus'])
        
        return {
            'primary': primary_target,
            'secondary': secondary_targets,
            'background': background_targets
        }
    
    def _get_attention_distribution(self) -> Dict[str, float]:
        """Get current attention weight distribution"""
        
        if not self.attention_targets:
            return {}
        
        total_weight = sum(target['weight'] for target in self.attention_targets.values())
        
        if total_weight == 0:
            return {}
        
        distribution = {}
        for target_id, target_data in self.attention_targets.items():
            distribution[target_id] = target_data['weight'] / total_weight
        
        return distribution
    
    def focus_on_target(self, target_id: str, boost_factor: float = 2.0):
        """Manually focus attention on a specific target"""
        
        if target_id in self.attention_targets:
            self.attention_targets[target_id]['weight'] *= boost_factor
            self.attention_targets[target_id]['focus_episodes'] += 1
            
            # Record focused attention in history
            self.attention_history.append({
                'type': 'manual_focus',
                'target_id': target_id,
                'timestamp': time.time(),
                'boost_factor': boost_factor
            })
    
    def ignore_target(self, target_id: str):
        """Actively ignore a specific target"""
        
        if target_id in self.attention_targets:
            self.attention_targets[target_id]['weight'] *= 0.1
            
            # Add to distraction filters
            self.distraction_filters[target_id] = {
                'ignore_until': time.time() + 30.0,  # Ignore for 30 seconds
                'reason': 'manual_ignore'
            }
    
    def get_current_focus(self) -> Optional[str]:
        """Get the current primary focus target"""
        
        if not self.attention_targets:
            return None
        
        # Find target with highest weight
        max_weight = 0.0
        focus_target = None
        
        for target_id, target_data in self.attention_targets.items():
            if target_data['weight'] > max_weight:
                max_weight = target_data['weight']
                focus_target = target_id
        
        return focus_target if max_weight > self.relevance_threshold else None
    
    def shift_attention(self, new_stimuli: List[Dict[str, Any]], 
                       attention_shift_trigger: str = "new_stimulus") -> Dict[str, Any]:
        """Handle attention shifting when new stimuli appear"""
        
        # Process new stimuli
        attention_result = self.process_stimuli(new_stimuli)
        
        # Record attention shift
        self.attention_history.append({
            'type': 'attention_shift',
            'trigger': attention_shift_trigger,
            'timestamp': time.time(),
            'new_focus': attention_result['primary_focus'],
            'stimulus_count': len(new_stimuli)
        })
        
        return attention_result
    
    def get_attention_summary(self) -> Dict[str, Any]:
        """Get summary of attention system state"""
        
        return {
            'active_targets': len(self.attention_targets),
            'current_focus': self.get_current_focus(),
            'attention_history_length': len(self.attention_history),
            'distraction_filters': len(self.distraction_filters),
            'total_cumulative_attention': sum(
                target['cumulative_attention'] for target in self.attention_targets.values()
            ),
            'average_focus_time': np.mean([
                target['total_focus_time'] for target in self.attention_targets.values()
            ]) if self.attention_targets else 0.0
        }
    
    def get_attention_patterns(self) -> Dict[str, Any]:
        """Analyze attention patterns and tendencies"""
        
        if not self.attention_history:
            return {'patterns': [], 'insights': []}
        
        # Analyze attention shift frequency
        recent_shifts = [event for event in self.attention_history[-50:] 
                        if event['type'] == 'attention_shift']
        
        shift_frequency = len(recent_shifts) / 50.0 if len(self.attention_history) >= 50 else 0.0
        
        # Analyze focus persistence
        focus_episodes = []
        for target_data in self.attention_targets.values():
            if target_data['focus_episodes'] > 0:
                avg_focus_time = target_data['total_focus_time'] / target_data['focus_episodes']
                focus_episodes.append(avg_focus_time)
        
        avg_focus_persistence = np.mean(focus_episodes) if focus_episodes else 0.0
        
        patterns = {
            'attention_shift_frequency': shift_frequency,
            'average_focus_persistence': avg_focus_persistence,
            'most_attended_targets': sorted(
                [(tid, data['cumulative_attention']) for tid, data in self.attention_targets.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }
        
        insights = []
        if shift_frequency > 0.3:
            insights.append("High attention shifting - may indicate distractibility")
        if avg_focus_persistence < 1.0:
            insights.append("Short focus duration - may need attention training")
        if len(self.attention_targets) > self.max_attention_targets:
            insights.append("Attention overload - too many simultaneous targets")
        
        return {'patterns': patterns, 'insights': insights}
