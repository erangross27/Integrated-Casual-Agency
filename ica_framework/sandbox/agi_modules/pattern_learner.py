"""
Pattern Learner Module
Learns and recognizes patterns in data and behaviors
"""

import time
import numpy as np
from typing import Dict, List, Any, Tuple


class PatternLearner:
    """Learns patterns from observations and experiences"""
    
    def __init__(self):
        self.behavioral_patterns = {}
        self.temporal_patterns = {}
        self.spatial_patterns = {}
        self.sequence_patterns = {}
        
        # Pattern learning parameters
        self.pattern_threshold = 3  # Minimum occurrences to establish pattern
        self.confidence_threshold = 0.75
        self.max_pattern_length = 10
    
    def learn_patterns(self, observations: List[Dict[str, Any]], context: str = "general"):
        """Learn patterns from a sequence of observations"""
        
        new_patterns = []
        
        # Learn temporal patterns
        temporal_patterns = self._learn_temporal_patterns(observations, context)
        new_patterns.extend(temporal_patterns)
        
        # Learn behavioral patterns
        behavioral_patterns = self._learn_behavioral_patterns(observations, context)
        new_patterns.extend(behavioral_patterns)
        
        # Learn spatial patterns
        spatial_patterns = self._learn_spatial_patterns(observations, context)
        new_patterns.extend(spatial_patterns)
        
        # Learn sequence patterns
        sequence_patterns = self._learn_sequence_patterns(observations, context)
        new_patterns.extend(sequence_patterns)
        
        return new_patterns
    
    def _learn_temporal_patterns(self, observations: List[Dict[str, Any]], context: str):
        """Learn patterns that repeat over time"""
        
        if context not in self.temporal_patterns:
            self.temporal_patterns[context] = {}
        
        patterns = self.temporal_patterns[context]
        new_patterns = []
        
        # Always create a basic observation pattern to get learning started
        basic_pattern_key = f"basic_observation_{context}"
        if basic_pattern_key not in patterns and observations:
            patterns[basic_pattern_key] = {
                'type': 'basic_observation',
                'event_type': 'observation',
                'intervals': [],
                'average_interval': 1.0,
                'confidence': 0.5,
                'occurrences': 1
            }
            new_patterns.append(basic_pattern_key)
        
        # Extract time intervals between similar events
        event_types = {}
        for obs in observations:
            event_type = obs.get('type', 'unknown')
            timestamp = obs.get('timestamp', time.time())
            
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(timestamp)
        
        # Find temporal patterns in each event type
        for event_type, timestamps in event_types.items():
            if len(timestamps) >= self.pattern_threshold:
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = timestamps[i] - timestamps[i-1]
                    intervals.append(interval)
                
                if intervals:
                    pattern_key = f"{event_type}_temporal"
                    if pattern_key not in patterns:
                        patterns[pattern_key] = {
                            'type': 'temporal_repetition',
                            'event_type': event_type,
                            'intervals': [],
                            'average_interval': 0.0,
                            'confidence': 0.0,
                            'occurrences': 0
                        }
                        new_patterns.append(pattern_key)  # Add to new patterns list!
                    
                    pattern = patterns[pattern_key]
                    pattern['intervals'].extend(intervals)
                    pattern['average_interval'] = np.mean(pattern['intervals'])
                    pattern['occurrences'] = len(timestamps)
                    
                    # Calculate confidence based on interval consistency
                    if len(pattern['intervals']) > 1:
                        std_dev = np.std(pattern['intervals'])
                        mean_interval = pattern['average_interval']
                        if mean_interval > 0:
                            consistency = 1.0 - min(1.0, std_dev / mean_interval)
                            pattern['confidence'] = consistency
        
        return new_patterns
    
    def _learn_behavioral_patterns(self, observations: List[Dict[str, Any]], context: str):
        """Learn patterns in object behavior"""
        
        if context not in self.behavioral_patterns:
            self.behavioral_patterns[context] = {}
        
        patterns = self.behavioral_patterns[context]
        
        # Group observations by object
        object_behaviors = {}
        for obs in observations:
            obj_id = obs.get('object_id', 'unknown')
            if obj_id not in object_behaviors:
                object_behaviors[obj_id] = []
            object_behaviors[obj_id].append(obs)
        
        # Find behavioral patterns for each object
        for obj_id, behaviors in object_behaviors.items():
            if len(behaviors) >= self.pattern_threshold:
                # Look for repeated action sequences
                action_sequence = []
                for behavior in behaviors:
                    action = behavior.get('action', 'idle')
                    action_sequence.append(action)
                
                # Find repeated subsequences
                repeated_sequences = self._find_repeated_subsequences(action_sequence)
                
                for sequence, count in repeated_sequences.items():
                    if count >= self.pattern_threshold:
                        pattern_key = f"{obj_id}_behavior_{hash(sequence)}"
                        patterns[pattern_key] = {
                            'type': 'behavioral_sequence',
                            'object_id': obj_id,
                            'sequence': sequence,
                            'occurrences': count,
                            'confidence': min(1.0, count / len(behaviors)),
                            'context': context
                        }
        
        return []  # For now, return empty list
    
    def _learn_spatial_patterns(self, observations: List[Dict[str, Any]], context: str):
        """Learn patterns in spatial arrangements and movements"""
        
        if context not in self.spatial_patterns:
            self.spatial_patterns[context] = {}
        
        patterns = self.spatial_patterns[context]
        
        # Extract spatial information
        positions = []
        movements = []
        
        for obs in observations:
            position = obs.get('position', None)
            velocity = obs.get('velocity', None)
            
            if position is not None:
                positions.append(position)
            
            if velocity is not None and any(abs(v) > 0.1 for v in velocity):
                movement_direction = self._normalize_vector(velocity)
                movements.append(movement_direction)
        
        # Find spatial clustering patterns
        if len(positions) >= self.pattern_threshold:
            clusters = self._find_spatial_clusters(positions)
            
            for i, cluster in enumerate(clusters):
                if len(cluster) >= self.pattern_threshold:
                    pattern_key = f"spatial_cluster_{i}"
                    center = np.mean(cluster, axis=0)
                    
                    patterns[pattern_key] = {
                        'type': 'spatial_cluster',
                        'center': center.tolist(),
                        'positions': cluster,
                        'size': len(cluster),
                        'confidence': len(cluster) / len(positions),
                        'context': context
                    }
        
        # Find movement direction patterns
        if len(movements) >= self.pattern_threshold:
            common_directions = self._find_common_directions(movements)
            
            for direction, count in common_directions.items():
                if count >= self.pattern_threshold:
                    pattern_key = f"movement_direction_{hash(str(direction))}"
                    patterns[pattern_key] = {
                        'type': 'movement_pattern',
                        'direction': direction,
                        'occurrences': count,
                        'confidence': count / len(movements),
                        'context': context
                    }
        
        return []  # For now, return empty list
    
    def _learn_sequence_patterns(self, observations: List[Dict[str, Any]], context: str):
        """Learn patterns in sequences of events or states"""
        
        if context not in self.sequence_patterns:
            self.sequence_patterns[context] = {}
        
        patterns = self.sequence_patterns[context]
        
        # Extract event sequence
        event_sequence = []
        for obs in observations:
            event_type = obs.get('type', 'unknown')
            event_sequence.append(event_type)
        
        # Find common subsequences
        for length in range(2, min(self.max_pattern_length + 1, len(event_sequence))):
            subsequences = {}
            
            for i in range(len(event_sequence) - length + 1):
                subseq = tuple(event_sequence[i:i + length])
                if subseq not in subsequences:
                    subsequences[subseq] = 0
                subsequences[subseq] += 1
            
            # Store patterns that occur frequently enough
            for subseq, count in subsequences.items():
                if count >= self.pattern_threshold:
                    pattern_key = f"sequence_pattern_{hash(subseq)}"
                    patterns[pattern_key] = {
                        'type': 'event_sequence',
                        'sequence': list(subseq),
                        'length': length,
                        'occurrences': count,
                        'confidence': count / (len(event_sequence) - length + 1),
                        'context': context
                    }
        
        return []  # For now, return empty list
    
    def _find_repeated_subsequences(self, sequence: List[str]) -> Dict[str, int]:
        """Find repeated subsequences in a sequence"""
        
        repeated = {}
        
        for length in range(2, min(self.max_pattern_length + 1, len(sequence))):
            for i in range(len(sequence) - length + 1):
                subseq = tuple(sequence[i:i + length])
                subseq_str = str(subseq)
                
                if subseq_str not in repeated:
                    repeated[subseq_str] = 0
                repeated[subseq_str] += 1
        
        # Filter out patterns that occur only once
        return {k: v for k, v in repeated.items() if v > 1}
    
    def _normalize_vector(self, vector: List[float]) -> Tuple[float, float, float]:
        """Normalize a vector to unit length"""
        
        magnitude = np.linalg.norm(vector)
        if magnitude > 0:
            return tuple(np.array(vector) / magnitude)
        return (0.0, 0.0, 0.0)
    
    def _find_spatial_clusters(self, positions: List[List[float]]) -> List[List[List[float]]]:
        """Find spatial clusters in position data"""
        
        # Simple clustering algorithm
        clusters = []
        used_positions = set()
        cluster_threshold = 2.0  # Distance threshold for clustering
        
        for i, pos in enumerate(positions):
            if i in used_positions:
                continue
            
            cluster = [pos]
            used_positions.add(i)
            
            for j, other_pos in enumerate(positions):
                if j in used_positions:
                    continue
                
                distance = np.linalg.norm(np.array(pos) - np.array(other_pos))
                if distance < cluster_threshold:
                    cluster.append(other_pos)
                    used_positions.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _find_common_directions(self, movements: List[Tuple[float, float, float]]) -> Dict[Tuple[float, float, float], int]:
        """Find commonly used movement directions"""
        
        direction_counts = {}
        direction_threshold = 0.3  # Similarity threshold for grouping directions
        
        for movement in movements:
            # Find if this direction is similar to an existing one
            found_similar = False
            for existing_dir in direction_counts:
                similarity = np.dot(movement, existing_dir)
                if similarity > (1.0 - direction_threshold):
                    direction_counts[existing_dir] += 1
                    found_similar = True
                    break
            
            if not found_similar:
                direction_counts[movement] = 1
        
        return direction_counts
    
    def predict_next_event(self, recent_events: List[str], context: str = "general") -> Dict[str, Any]:
        """Predict the next event based on learned patterns"""
        
        if context not in self.sequence_patterns:
            return None
        
        patterns = self.sequence_patterns[context]
        
        # Look for patterns that match the end of recent_events
        best_prediction = None
        best_confidence = 0.0
        
        for pattern_key, pattern_data in patterns.items():
            if pattern_data['type'] == 'event_sequence':
                sequence = pattern_data['sequence']
                pattern_confidence = pattern_data['confidence']
                
                # Check if recent events match the beginning of this pattern
                if len(recent_events) >= len(sequence) - 1:
                    match_sequence = sequence[:-1]  # All but the last element
                    recent_tail = recent_events[-(len(match_sequence)):]
                    
                    if recent_tail == match_sequence and pattern_confidence > best_confidence:
                        best_prediction = {
                            'predicted_event': sequence[-1],
                            'confidence': pattern_confidence,
                            'pattern_key': pattern_key,
                            'context': context
                        }
                        best_confidence = pattern_confidence
        
        return best_prediction
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all learned patterns"""
        
        summary = {
            'temporal_patterns': len(self.temporal_patterns),
            'behavioral_patterns': len(self.behavioral_patterns),
            'spatial_patterns': len(self.spatial_patterns),
            'sequence_patterns': len(self.sequence_patterns),
            'high_confidence_patterns': 0
        }
        
        # Count high-confidence patterns
        all_patterns = []
        for context_patterns in [self.temporal_patterns, self.behavioral_patterns, 
                               self.spatial_patterns, self.sequence_patterns]:
            for context in context_patterns.values():
                all_patterns.extend(context.values())
        
        for pattern in all_patterns:
            if pattern.get('confidence', 0) > self.confidence_threshold:
                summary['high_confidence_patterns'] += 1
        
        return summary
