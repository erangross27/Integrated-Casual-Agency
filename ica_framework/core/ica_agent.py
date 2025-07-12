"""
Main ICA Agent implementation
Integrates all components for the complete ICA framework
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from pathlib import Path

from ..components import (
    CausalKnowledgeGraph, Node, Edge,
    WorldModel, WorldModelTrainer,
    CuriosityModule,
    ActionPlanner,
    HierarchicalAbstraction
)
from ..utils import Config, ica_logger, Metrics, Visualizer


@dataclass
class ExperimentResult:
    """Result of an experiment/action"""
    state_before: np.ndarray
    action: np.ndarray
    state_after: np.ndarray
    reward: float
    done: bool
    prediction_error: float
    confidence_change: float
    timestamp: float


class ICAAgent:
    """
    Main Integrated Causal Agency Agent
    
    Implements the complete ICA framework with:
    - Causal Knowledge Graph for world representation
    - Bayesian World Model for predictions
    - Curiosity Module for intrinsic motivation
    - Action Planner for decision making
    - Hierarchical Abstraction for concept formation
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = ica_logger
        self.device = torch.device(config.get_device())
        
        # Initialize components
        self.knowledge_graph = CausalKnowledgeGraph(config.graph)
        self.world_model = None  # Will be initialized when graph structure is known
        self.curiosity_module = None  # Will be initialized with state/action dimensions
        self.action_planner = None  # Will be initialized with state/action dimensions
        self.hierarchical_abstraction = HierarchicalAbstraction(config.abstraction)
        
        # Utilities
        self.metrics = Metrics()
        self.visualizer = Visualizer()
        
        # Agent state
        self.current_state = None
        self.global_confidence = 0.5  # Initial global confidence
        self.step_count = 0
        self.episode_count = 0
        self.experiment_history = []
        
        # Learning parameters
        self.alpha = config.abstraction.significance_threshold  # Significance threshold
        self.gamma = config.abstraction.utility_decay  # Utility decay factor
        
        self.logger.info(f"Initialized ICA Agent with config on device: {self.device}")
    
    def initialize_world_model(self, state_dim: int, action_dim: int, num_relations: int):
        """Initialize world model with known dimensions"""
        
        self.world_model = WorldModel(
            self.config.world_model,
            num_relations=num_relations,
            num_node_features=state_dim
        ).to(self.device)
        
        self.world_model_trainer = WorldModelTrainer(self.world_model, self.config.world_model)
        
        # Initialize curiosity module
        self.curiosity_module = CuriosityModule(
            self.config.curiosity,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Initialize action planner
        self.action_planner = ActionPlanner(
            self.config.planner,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        self.logger.info(f"Initialized world model with state_dim={state_dim}, action_dim={action_dim}")
    
    def observe_environment(self, observation: Dict[str, Any]) -> bool:
        """
        Process environmental observation and update knowledge graph
        
        Args:
            observation: Dictionary containing state, entities, relationships, etc.
            
        Returns:
            Success flag
        """
        
        try:
            # Extract entities and relationships from observation
            entities = observation.get('entities', [])
            relationships = observation.get('relationships', [])
            
            # Update knowledge graph
            for entity in entities:
                node = Node(
                    id=entity['id'],
                    label=entity.get('label', 'unknown'),
                    properties_static=entity.get('properties_static', {}),
                    properties_dynamic=entity.get('properties_dynamic', {}),
                    confidence=entity.get('confidence', 1.0)
                )
                self.knowledge_graph.add_node(node)
            
            for relationship in relationships:
                edge = Edge(
                    source=relationship['source'],
                    target=relationship['target'],
                    relationship=relationship['type'],
                    properties=relationship.get('properties', {}),
                    confidence=relationship.get('confidence', 0.5),
                    weight=relationship.get('weight', 1.0),
                    conditions=relationship.get('conditions', {})
                )
                self.knowledge_graph.add_edge(edge)
            
            # Update current state
            self.current_state = observation.get('state', self.current_state)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error observing environment: {str(e)}")
            return False
    
    def formulate_hypothesis(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formulate hypothesis based on observed anomaly
        
        Args:
            anomaly: Description of the anomaly that triggered curiosity
            
        Returns:
            Hypothesis dictionary
        """
        
        # Analyze anomaly
        anomaly_type = anomaly.get('type', 'unknown')
        affected_entities = anomaly.get('entities', [])
        
        # Find related nodes in knowledge graph
        related_nodes = []
        for entity_id in affected_entities:
            neighbors = self.knowledge_graph.get_neighbors(entity_id)
            related_nodes.extend(neighbors)
        
        # Generate hypothesis based on graph structure
        hypothesis = {
            'id': f"hyp_{self.step_count}_{time.time()}",
            'type': f"causal_explanation_{anomaly_type}",
            'entities': affected_entities + related_nodes,
            'predicted_relationships': [],
            'confidence': 0.3,  # Initial low confidence
            'created_at': time.time()
        }
        
        # Use hierarchical abstraction to inform hypothesis
        if self.hierarchical_abstraction.current_concepts:
            relevant_concepts = self._find_relevant_concepts(affected_entities)
            hypothesis['relevant_concepts'] = relevant_concepts
        
        self.logger.debug(f"Formulated hypothesis: {hypothesis['id']}")
        return hypothesis
    
    def design_experiment(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design experiment to test hypothesis
        
        Args:
            hypothesis: Hypothesis to test
            
        Returns:
            Experiment design dictionary
        """
        
        # Determine what action would best test the hypothesis
        target_entities = hypothesis.get('entities', [])
        
        # Use action planner to suggest action
        if self.action_planner and self.current_state is not None:
            # Convert current state to tensor
            state_tensor = torch.FloatTensor(self.current_state)
            
            # Get action from planner
            action = self.action_planner.select_action(self.current_state)
        else:
            # Random action if planner not available
            action = np.random.normal(0, 0.1, size=4)  # Default action size
        
        experiment = {
            'id': f"exp_{self.step_count}_{time.time()}",
            'hypothesis_id': hypothesis['id'],
            'action': action,
            'target_entities': target_entities,
            'expected_outcome': hypothesis.get('predicted_relationships', []),
            'designed_at': time.time()
        }
        
        self.logger.debug(f"Designed experiment: {experiment['id']}")
        return experiment
    
    def execute_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Execute action in the environment (placeholder for actual environment interaction)
        
        Args:
            action: Action to execute
            
        Returns:
            Observation from environment
        """
        
        # This is a placeholder - in real implementation, this would interact with environment
        # For now, simulate a response
        
        # Simulate state change
        if self.current_state is not None:
            noise = np.random.normal(0, 0.01, size=self.current_state.shape)
            # Handle dimension mismatch between action and state
            if len(action) < len(self.current_state):
                # Pad action with zeros if it's smaller than state
                action_padded = np.zeros(len(self.current_state))
                action_padded[:len(action)] = action
                next_state = self.current_state + action_padded * 0.1 + noise
            elif len(action) > len(self.current_state):
                # Truncate action if it's larger than state
                next_state = self.current_state + action[:len(self.current_state)] * 0.1 + noise
            else:
                # Same dimensions
                next_state = self.current_state + action * 0.1 + noise
        else:
            next_state = np.random.normal(0, 0.1, size=10)  # Default state size
        
        # Simulate reward
        reward = np.random.normal(0, 0.1)
        
        # Simulate done flag
        done = False
        
        observation = {
            'state': next_state,
            'reward': reward,
            'done': done,
            'entities': [],  # Would be populated by environment
            'relationships': []  # Would be populated by environment
        }
        
        return observation
    
    def update_knowledge(self, experiment: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """
        Update knowledge graph based on experiment results
        
        Args:
            experiment: Experiment that was executed
            result: Result of the experiment
            
        Returns:
            Success flag
        """
        
        try:
            # Calculate prediction error
            if self.current_state is not None and self.world_model is not None:
                # This is simplified - in practice would use actual world model prediction
                prediction_error = np.random.normal(0, 0.1)  # Placeholder
            else:
                prediction_error = 0.0
            
            # Update confidence based on prediction accuracy
            confidence_change = -abs(prediction_error) * 0.1  # Decrease confidence if high error
            
            # Update knowledge graph edges based on results
            target_entities = experiment.get('target_entities', [])
            for i, entity1 in enumerate(target_entities):
                for entity2 in target_entities[i+1:]:
                    # Find edges between entities
                    try:
                        # Update edge confidence based on experiment outcome
                        edge_id = (entity1, entity2, 0)  # Simplified edge ID
                        success = prediction_error < 0.1  # Define success threshold
                        self.knowledge_graph.update_edge_confidence(edge_id, success)
                    except:
                        pass  # Edge doesn't exist
            
            # Store experiment result
            experiment_result = ExperimentResult(
                state_before=self.current_state.copy() if self.current_state is not None else np.array([]),
                action=experiment['action'],
                state_after=result['state'],
                reward=result['reward'],
                done=result['done'],
                prediction_error=prediction_error,
                confidence_change=confidence_change,
                timestamp=time.time()
            )
            
            self.experiment_history.append(experiment_result)
            
            # Update global confidence
            confidence_change_value = confidence_change.item() if hasattr(confidence_change, 'item') else float(confidence_change)
            self.global_confidence = max(0.0, min(1.0, self.global_confidence + confidence_change_value))
            
            self.logger.debug(f"Updated knowledge, global confidence: {self.global_confidence:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating knowledge: {str(e)}")
            return False
    
    def active_learning_step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one step of active learning (observe -> hypothesize -> experiment -> update)
        
        Args:
            observation: Current environmental observation
            
        Returns:
            Dictionary containing step results
        """
        
        step_start = time.time()
        
        # 1. Observe environment
        self.observe_environment(observation)
        
        # 2. Detect anomalies (simplified)
        anomalies = self._detect_anomalies(observation)
        
        # 3. Process each anomaly
        step_results = {
            'step': self.step_count,
            'anomalies_detected': len(anomalies),
            'experiments_conducted': 0,
            'knowledge_updates': 0,
            'global_confidence': self.global_confidence,
            'intrinsic_reward': 0.0,
            'actions_taken': []
        }
        
        for anomaly in anomalies:
            # Formulate hypothesis
            hypothesis = self.formulate_hypothesis(anomaly)
            
            # Design experiment
            experiment = self.design_experiment(hypothesis)
            
            # Execute action
            action_result = self.execute_action(experiment['action'])
            
            # Update knowledge
            if self.update_knowledge(experiment, action_result):
                step_results['knowledge_updates'] += 1
            
            step_results['experiments_conducted'] += 1
            step_results['actions_taken'].append(experiment['action'])
        
        # 4. Calculate intrinsic reward
        if self.curiosity_module and len(step_results['actions_taken']) > 0:
            # Simplified intrinsic reward calculation
            avg_prediction_error = np.mean([exp.prediction_error for exp in self.experiment_history[-5:]])
            intrinsic_reward = self.curiosity_module.calculate_intrinsic_reward(
                torch.tensor(avg_prediction_error),
                torch.randn(128),  # Placeholder model representation (matches complexity estimator input size)
                torch.randn(128)   # Placeholder updated model representation (matches complexity estimator input size)
            )
            step_results['intrinsic_reward'] = intrinsic_reward.item()
        
        # 5. Update action planner
        if self.action_planner and self.current_state is not None:
            # Ensure global_confidence is a float, not a tensor
            confidence_value = self.global_confidence.item() if hasattr(self.global_confidence, 'item') else float(self.global_confidence)
            self.action_planner.update_alpha(confidence_value)
        
        # 6. Update hierarchical abstraction
        if self.knowledge_graph.graph.number_of_nodes() > 0:
            self.logger.debug(f"Processing graph with {self.knowledge_graph.graph.number_of_nodes()} nodes and {self.knowledge_graph.graph.number_of_edges()} edges")
            abstraction_results = self.hierarchical_abstraction.process_graph(
                self.knowledge_graph.graph
            )
            step_results['abstraction_results'] = abstraction_results
        else:
            self.logger.debug("Skipping hierarchical abstraction - knowledge graph is empty")
        
        # 7. Update metrics
        confidence_value = self.global_confidence.item() if hasattr(self.global_confidence, 'item') else float(self.global_confidence)
        self.metrics.update_metrics('global_confidence', confidence_value, self.step_count)
        self.metrics.update_metrics('intrinsic_reward', step_results['intrinsic_reward'], self.step_count)
        
        # Update step count
        self.step_count += 1
        
        step_duration = time.time() - step_start
        confidence_value = self.global_confidence.item() if hasattr(self.global_confidence, 'item') else float(self.global_confidence)
        self.logger.info(f"Step {self.step_count} completed in {step_duration:.3f}s: "
                        f"{step_results['experiments_conducted']} experiments, "
                        f"confidence: {confidence_value:.3f}")
        
        return step_results
    
    def _detect_anomalies(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in observation that warrant investigation"""
        
        anomalies = []
        
        # Simple anomaly detection based on state change
        if self.current_state is not None:
            new_state = observation.get('state', self.current_state)
            state_change = np.linalg.norm(new_state - self.current_state)
            
            if state_change > 0.1:  # Threshold for significant change
                anomalies.append({
                    'type': 'state_change',
                    'magnitude': state_change,
                    'entities': observation.get('entities', []),
                    'description': f"Significant state change detected: {state_change:.3f}"
                })
        
        # Check for new entities
        current_entities = set(self.knowledge_graph.nodes_dict.keys())
        new_entities = set(e['id'] for e in observation.get('entities', []))
        
        if new_entities - current_entities:
            anomalies.append({
                'type': 'new_entities',
                'entities': list(new_entities - current_entities),
                'description': f"New entities detected: {list(new_entities - current_entities)}"
            })
        
        return anomalies
    
    def _find_relevant_concepts(self, entities: List[str]) -> List[str]:
        """Find concepts relevant to given entities"""
        
        relevant_concepts = []
        
        for concept_name, concept_data in self.hierarchical_abstraction.current_concepts.items():
            # Check if any concept motifs contain the entities
            for motif_id in concept_data['motifs']:
                motif_entities = eval(motif_id)  # Convert string back to list
                if any(entity in motif_entities for entity in entities):
                    relevant_concepts.append(concept_name)
                    break
        
        return relevant_concepts
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Get comprehensive agent state"""
        
        # Get basic agent state
        state = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'global_confidence': float(self.global_confidence),
            'current_state': self.current_state.tolist() if self.current_state is not None else None,
            'experiment_history_length': len(self.experiment_history)
        }
        
        # Get graph stats safely
        try:
            graph_stats = self.knowledge_graph.get_graph_stats()
            # Ensure all values are JSON serializable
            serializable_stats = {}
            for key, value in graph_stats.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_stats[key] = float(value)
                elif isinstance(value, dict):
                    # Handle nested dicts (like centrality_stats)
                    nested_dict = {}
                    for k, v in value.items():
                        if isinstance(v, (np.integer, np.floating)):
                            nested_dict[k] = float(v)
                        else:
                            nested_dict[k] = v
                    serializable_stats[key] = nested_dict
                else:
                    serializable_stats[key] = value
            state['graph_stats'] = serializable_stats
        except Exception as e:
            self.logger.warning(f"Error getting graph stats: {e}")
            state['graph_stats'] = {
                'num_nodes': self.knowledge_graph.graph.number_of_nodes(),
                'num_edges': self.knowledge_graph.graph.number_of_edges()
            }
        
        # Get metrics safely
        try:
            metrics = self.metrics.get_metrics_summary()
            # Convert numpy types to Python types
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = float(value)
                elif isinstance(value, (list, tuple)):
                    serializable_metrics[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
                else:
                    serializable_metrics[key] = value
            state['metrics'] = serializable_metrics
        except Exception as e:
            self.logger.warning(f"Error getting metrics: {e}")
            state['metrics'] = {}
        
        # Add component-specific metrics safely
        try:
            if self.curiosity_module:
                curiosity_metrics = self.curiosity_module.get_curiosity_metrics()
                state['curiosity_metrics'] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                            for k, v in curiosity_metrics.items()}
        except Exception as e:
            self.logger.warning(f"Error getting curiosity metrics: {e}")
            state['curiosity_metrics'] = {}
        
        try:
            if self.action_planner:
                planner_metrics = self.action_planner.get_planner_metrics()
                state['planner_metrics'] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                          for k, v in planner_metrics.items()}
        except Exception as e:
            self.logger.warning(f"Error getting planner metrics: {e}")
            state['planner_metrics'] = {}
        
        try:
            abstraction_metrics = self.hierarchical_abstraction.get_abstraction_metrics()
            state['abstraction_metrics'] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                          for k, v in abstraction_metrics.items()}
        except Exception as e:
            self.logger.warning(f"Error getting abstraction metrics: {e}")
            state['abstraction_metrics'] = {}
        
        return state
    
    def save_agent(self, directory: str) -> bool:
        """Save complete agent state"""
        
        try:
            save_dir = Path(directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save knowledge graph
            self.knowledge_graph.save(str(save_dir / "knowledge_graph.json"))
            
            # Save world model
            if self.world_model and self.world_model_trainer:
                self.world_model_trainer.save_checkpoint(
                    str(save_dir / "world_model.pth"), 
                    self.step_count, 
                    self.metrics.get_metrics_summary()
                )
            
            # Save curiosity module
            if self.curiosity_module:
                self.curiosity_module.save_state(str(save_dir / "curiosity_module.pth"))
            
            # Save action planner
            if self.action_planner:
                self.action_planner.save_checkpoint(str(save_dir / "action_planner.pth"))
            
            # Save hierarchical abstraction
            self.hierarchical_abstraction.save_state(str(save_dir / "hierarchical_abstraction.json"))
            
            # Save agent state
            agent_state = self.get_agent_state()
            
            # Make agent state JSON serializable
            def make_serializable(obj):
                """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
                import networkx as nx
                
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                    # Convert NetworkX graph to serializable format
                    return {
                        'nodes': list(obj.nodes(data=True)),
                        'edges': list(obj.edges(data=True)),
                        'graph_type': obj.__class__.__name__,
                        'num_nodes': obj.number_of_nodes(),
                        'num_edges': obj.number_of_edges()
                    }
                else:
                    return obj
            
            serializable_state = make_serializable(agent_state)
            
            import json
            with open(save_dir / "agent_state.json", 'w') as f:
                json.dump(serializable_state, f, indent=2)
            
            self.logger.info(f"Saved agent to {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving agent: {str(e)}")
            return False
    
    def load_agent(self, directory: str) -> bool:
        """Load complete agent state"""
        
        try:
            load_dir = Path(directory)
            
            # Load knowledge graph
            self.knowledge_graph.load(str(load_dir / "knowledge_graph.json"))
            
            # Load hierarchical abstraction
            self.hierarchical_abstraction.load_state(str(load_dir / "hierarchical_abstraction.json"))
            
            # Load agent state
            import json
            with open(load_dir / "agent_state.json", 'r') as f:
                agent_state = json.load(f)
            
            self.step_count = agent_state['step_count']
            self.episode_count = agent_state['episode_count']
            self.global_confidence = agent_state['global_confidence']
            
            if agent_state['current_state']:
                self.current_state = np.array(agent_state['current_state'])
            
            # Load other components if they exist
            if (load_dir / "curiosity_module.pth").exists() and self.curiosity_module:
                self.curiosity_module.load_state(str(load_dir / "curiosity_module.pth"))
            
            if (load_dir / "action_planner.pth").exists() and self.action_planner:
                self.action_planner.load_checkpoint(str(load_dir / "action_planner.pth"))
            
            self.logger.info(f"Loaded agent from {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading agent: {str(e)}")
            return False
