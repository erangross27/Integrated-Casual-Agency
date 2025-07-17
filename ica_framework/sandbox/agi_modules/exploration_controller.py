"""
Exploration Controller Module
Controls exploration behavior and goal-directed activity
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class ExplorationController:
    """Controls exploration strategies and goal-directed behavior"""
    
    def __init__(self):
        self.exploration_history = []
        self.explored_areas = set()
        self.current_goals = []
        self.exploration_strategies = {}
        
        # Exploration parameters
        self.exploration_radius = 5.0
        self.novelty_threshold = 0.3
        self.goal_achievement_radius = 1.0
        self.max_goals = 5
        
        # Initialize exploration strategies
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize different exploration strategies"""
        
        self.exploration_strategies = {
            'random_walk': {
                'name': 'Random Walk',
                'effectiveness': 0.5,
                'usage_count': 0,
                'description': 'Move randomly to explore unknown areas'
            },
            'gradient_ascent': {
                'name': 'Gradient Ascent',
                'effectiveness': 0.7,
                'usage_count': 0,
                'description': 'Move toward areas of high interest/novelty'
            },
            'systematic_grid': {
                'name': 'Systematic Grid',
                'effectiveness': 0.8,
                'usage_count': 0,
                'description': 'Systematically explore in a grid pattern'
            },
            'curiosity_driven': {
                'name': 'Curiosity Driven',
                'effectiveness': 0.6,
                'usage_count': 0,
                'description': 'Explore areas that generate questions'
            },
            'goal_directed': {
                'name': 'Goal Directed',
                'effectiveness': 0.9,
                'usage_count': 0,
                'description': 'Move toward specific objectives'
            }
        }
    
    def plan_exploration(self, current_position: List[float], 
                        environment_info: Dict[str, Any],
                        curiosity_targets: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Plan the next exploration action"""
        
        # Assess current situation
        novelty_score = self._assess_area_novelty(current_position)
        goal_priority = self._assess_goal_priority()
        
        # Choose exploration strategy
        strategy = self._choose_exploration_strategy(novelty_score, goal_priority, curiosity_targets)
        
        # Generate exploration action based on strategy
        action = self._generate_exploration_action(strategy, current_position, 
                                                 environment_info, curiosity_targets)
        
        # Record exploration decision
        self._record_exploration(current_position, strategy, action)
        
        return action
    
    def _assess_area_novelty(self, position: List[float]) -> float:
        """Assess how novel/unexplored the current area is"""
        
        # Check how close we are to previously explored areas
        min_distance = float('inf')
        
        for explored_pos in self.explored_areas:
            distance = np.linalg.norm(np.array(position) - np.array(explored_pos))
            min_distance = min(min_distance, distance)
        
        if min_distance == float('inf'):
            return 1.0  # Completely unexplored
        
        # Higher novelty for areas farther from explored regions
        novelty = min(1.0, min_distance / self.exploration_radius)
        return novelty
    
    def _assess_goal_priority(self) -> float:
        """Assess the priority of current goals"""
        
        if not self.current_goals:
            return 0.0
        
        # Calculate average priority of current goals
        total_priority = sum(goal.get('priority', 0.5) for goal in self.current_goals)
        return total_priority / len(self.current_goals)
    
    def _choose_exploration_strategy(self, novelty_score: float, goal_priority: float,
                                   curiosity_targets: List[Dict[str, Any]] = None) -> str:
        """Choose the best exploration strategy for current situation"""
        
        # If we have high-priority goals, use goal-directed exploration
        if goal_priority > 0.8:
            return 'goal_directed'
        
        # If we have curiosity targets, use curiosity-driven exploration
        if curiosity_targets and len(curiosity_targets) > 0:
            return 'curiosity_driven'
        
        # If in a novel area, use systematic exploration
        if novelty_score > 0.7:
            return 'systematic_grid'
        
        # If in moderately explored area, use gradient ascent toward novelty
        if novelty_score > 0.3:
            return 'gradient_ascent'
        
        # Default to random walk in well-explored areas
        return 'random_walk'
    
    def _generate_exploration_action(self, strategy: str, current_position: List[float],
                                   environment_info: Dict[str, Any],
                                   curiosity_targets: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate specific exploration action based on strategy"""
        
        if strategy == 'random_walk':
            return self._random_walk_action(current_position)
        
        elif strategy == 'gradient_ascent':
            return self._gradient_ascent_action(current_position)
        
        elif strategy == 'systematic_grid':
            return self._systematic_grid_action(current_position)
        
        elif strategy == 'curiosity_driven':
            return self._curiosity_driven_action(current_position, curiosity_targets)
        
        elif strategy == 'goal_directed':
            return self._goal_directed_action(current_position)
        
        else:
            return self._random_walk_action(current_position)  # Fallback
    
    def _random_walk_action(self, current_position: List[float]) -> Dict[str, Any]:
        """Generate random walk exploration action"""
        
        # Random direction
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(1.0, 3.0)
        
        target_position = [
            current_position[0] + distance * np.cos(angle),
            current_position[1],  # Keep same height
            current_position[2] + distance * np.sin(angle)
        ]
        
        return {
            'type': 'move_to_position',
            'target_position': target_position,
            'strategy': 'random_walk',
            'priority': 0.3,
            'estimated_duration': distance / 2.0  # Assuming speed of 2 units/sec
        }
    
    def _gradient_ascent_action(self, current_position: List[float]) -> Dict[str, Any]:
        """Move toward areas of highest novelty"""
        
        # Find direction with highest novelty gradient
        best_direction = None
        best_novelty = 0.0
        
        # Sample directions around current position
        for angle in np.linspace(0, 2 * np.pi, 8):
            test_position = [
                current_position[0] + 2.0 * np.cos(angle),
                current_position[1],
                current_position[2] + 2.0 * np.sin(angle)
            ]
            
            novelty = self._assess_area_novelty(test_position)
            if novelty > best_novelty:
                best_novelty = novelty
                best_direction = test_position
        
        if best_direction is None:
            return self._random_walk_action(current_position)
        
        return {
            'type': 'move_to_position',
            'target_position': best_direction,
            'strategy': 'gradient_ascent',
            'priority': 0.6,
            'estimated_duration': 1.0
        }
    
    def _systematic_grid_action(self, current_position: List[float]) -> Dict[str, Any]:
        """Systematically explore in a grid pattern"""
        
        # Define grid spacing
        grid_spacing = 3.0
        
        # Find nearest unexplored grid point
        grid_x = round(current_position[0] / grid_spacing) * grid_spacing
        grid_z = round(current_position[2] / grid_spacing) * grid_spacing
        
        # Check surrounding grid points
        grid_candidates = [
            [grid_x + grid_spacing, current_position[1], grid_z],
            [grid_x - grid_spacing, current_position[1], grid_z],
            [grid_x, current_position[1], grid_z + grid_spacing],
            [grid_x, current_position[1], grid_z - grid_spacing]
        ]
        
        # Choose the grid point with highest novelty
        best_candidate = None
        best_novelty = 0.0
        
        for candidate in grid_candidates:
            novelty = self._assess_area_novelty(candidate)
            if novelty > best_novelty:
                best_novelty = novelty
                best_candidate = candidate
        
        if best_candidate is None:
            best_candidate = grid_candidates[0]  # Fallback
        
        return {
            'type': 'move_to_position',
            'target_position': best_candidate,
            'strategy': 'systematic_grid',
            'priority': 0.7,
            'estimated_duration': 1.5
        }
    
    def _curiosity_driven_action(self, current_position: List[float],
                               curiosity_targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Move toward objects or areas that sparked curiosity"""
        
        if not curiosity_targets:
            return self._random_walk_action(current_position)
        
        # Choose the closest high-interest curiosity target
        best_target = None
        best_score = 0.0
        
        for target in curiosity_targets:
            target_position = target.get('position', [0, 0, 0])
            curiosity_level = target.get('curiosity_level', 0.0)
            
            distance = np.linalg.norm(np.array(current_position) - np.array(target_position))
            
            # Score combines curiosity level with proximity bias
            score = curiosity_level * (1.0 / (1.0 + distance * 0.1))
            
            if score > best_score:
                best_score = score
                best_target = target
        
        if best_target is None:
            return self._random_walk_action(current_position)
        
        return {
            'type': 'investigate_object',
            'target_position': best_target['position'],
            'target_object': best_target.get('object_id', 'unknown'),
            'strategy': 'curiosity_driven',
            'priority': 0.8,
            'estimated_duration': 2.0
        }
    
    def _goal_directed_action(self, current_position: List[float]) -> Dict[str, Any]:
        """Move toward the highest priority goal"""
        
        if not self.current_goals:
            return self._random_walk_action(current_position)
        
        # Sort goals by priority
        sorted_goals = sorted(self.current_goals, 
                            key=lambda g: g.get('priority', 0.0), 
                            reverse=True)
        
        top_goal = sorted_goals[0]
        goal_position = top_goal.get('position', current_position)
        
        return {
            'type': 'achieve_goal',
            'target_position': goal_position,
            'goal_id': top_goal.get('id', 'unknown'),
            'goal_description': top_goal.get('description', 'Unknown goal'),
            'strategy': 'goal_directed',
            'priority': top_goal.get('priority', 0.5),
            'estimated_duration': top_goal.get('estimated_duration', 3.0)
        }
    
    def _record_exploration(self, position: List[float], strategy: str, action: Dict[str, Any]):
        """Record the exploration decision and update history"""
        
        # Add to explored areas
        self.explored_areas.add(tuple(position))
        
        # Update strategy usage
        if strategy in self.exploration_strategies:
            self.exploration_strategies[strategy]['usage_count'] += 1
        
        # Record in history
        exploration_record = {
            'timestamp': time.time(),
            'position': position.copy(),
            'strategy': strategy,
            'action': action.copy(),
            'area_novelty': self._assess_area_novelty(position)
        }
        
        self.exploration_history.append(exploration_record)
        
        # Limit history size
        if len(self.exploration_history) > 1000:
            self.exploration_history = self.exploration_history[-500:]
    
    def add_goal(self, goal: Dict[str, Any]):
        """Add a new exploration goal"""
        
        if len(self.current_goals) >= self.max_goals:
            # Remove lowest priority goal
            self.current_goals.sort(key=lambda g: g.get('priority', 0.0))
            self.current_goals.pop(0)
        
        goal['timestamp'] = time.time()
        goal['id'] = f"goal_{len(self.exploration_history)}_{int(time.time())}"
        
        self.current_goals.append(goal)
    
    def update_goal_progress(self, current_position: List[float]) -> List[str]:
        """Check if any goals have been achieved and update progress"""
        
        completed_goals = []
        remaining_goals = []
        
        for goal in self.current_goals:
            goal_position = goal.get('position', [0, 0, 0])
            distance = np.linalg.norm(np.array(current_position) - np.array(goal_position))
            
            if distance <= self.goal_achievement_radius:
                completed_goals.append(goal['id'])
                
                # Update strategy effectiveness
                strategy = goal.get('strategy', 'unknown')
                if strategy in self.exploration_strategies:
                    self.exploration_strategies[strategy]['effectiveness'] *= 1.1
                    self.exploration_strategies[strategy]['effectiveness'] = min(1.0, 
                        self.exploration_strategies[strategy]['effectiveness'])
            else:
                remaining_goals.append(goal)
        
        self.current_goals = remaining_goals
        return completed_goals
    
    def get_exploration_summary(self) -> Dict[str, Any]:
        """Get summary of exploration progress and effectiveness"""
        
        return {
            'areas_explored': len(self.explored_areas),
            'total_exploration_actions': len(self.exploration_history),
            'current_goals': len(self.current_goals),
            'strategy_usage': {k: v['usage_count'] for k, v in self.exploration_strategies.items()},
            'strategy_effectiveness': {k: v['effectiveness'] for k, v in self.exploration_strategies.items()},
            'exploration_efficiency': self._calculate_exploration_efficiency()
        }
    
    def _calculate_exploration_efficiency(self) -> float:
        """Calculate overall exploration efficiency"""
        
        if not self.exploration_history:
            return 0.0
        
        # Measure how much new area was covered per action
        total_actions = len(self.exploration_history)
        unique_areas = len(self.explored_areas)
        
        efficiency = unique_areas / total_actions if total_actions > 0 else 0.0
        return min(1.0, efficiency)
    
    def get_recommended_action(self, current_position: List[float],
                             environment_info: Dict[str, Any],
                             curiosity_targets: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get the recommended exploration action for current situation"""
        
        return self.plan_exploration(current_position, environment_info, curiosity_targets)
