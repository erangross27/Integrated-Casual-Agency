#!/usr/bin/env python3
"""
Continuous Learning Agent

This script runs the ICA agent in a continuous learning simulation where the agent
persistently learns from its surrounding environment, building causal knowledge
and improving its understanding over time.
"""

import os
import time
import json
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Configure logging for continuous operation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from ica_framework import ICAAgent, Config
from ica_framework.sandbox import SandboxEnvironment


class ContinuousLearningEnvironment:
    """
    A dynamic simulation environment that provides continuous learning opportunities
    for the ICA agent through procedurally generated scenarios and challenges.
    """
    
    def __init__(self, config: Config, save_dir: str = "continuous_learning_data"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize agent
        self.agent = ICAAgent(config)
        
        # Simulation state
        self.step_count = 0
        self.session_start = datetime.now()
        self.last_save = datetime.now()
        self.save_interval = timedelta(minutes=10)  # Save every 10 minutes
        
        # Environment dynamics
        self.entities = self._initialize_entities()
        self.relations = self._initialize_relations()
        self.environment_complexity = 1.0
        self.complexity_growth_rate = 0.001
        
        # Learning metrics
        self.learning_history = []
        self.performance_metrics = {
            'total_experiments': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'confidence_trend': [],
            'knowledge_graph_size': [],
            'curiosity_rewards': []
        }
        
        logger.info(f"Continuous learning environment initialized in {save_dir}")
    
    def _initialize_entities(self) -> List[str]:
        """Initialize a set of entities for the simulation world."""
        entities = [
            # Physical objects
            'sphere_red', 'sphere_blue', 'cube_green', 'cube_yellow',
            'cylinder_orange', 'pyramid_purple', 'platform_gray',
            
            # Abstract concepts
            'gravity', 'friction', 'temperature', 'pressure',
            'energy', 'momentum', 'acceleration',
            
            # Agents/actors
            'robot_arm', 'conveyor_belt', 'sensor_array',
            'human_operator', 'automated_system',
            
            # Environmental factors
            'lighting_system', 'weather_conditions', 'magnetic_field',
            'electric_field', 'chemical_solution'
        ]
        return entities
    
    def _initialize_relations(self) -> List[str]:
        """Initialize possible relations between entities."""
        relations = [
            'causes', 'affects', 'influences', 'prevents', 'enables',
            'contains', 'supports', 'pushes', 'pulls', 'heats',
            'cools', 'accelerates', 'decelerates', 'transforms',
            'creates', 'destroys', 'connects', 'separates',
            'increases', 'decreases', 'stabilizes', 'destabilizes'
        ]
        return relations
    
    def generate_observation(self) -> Dict[str, Any]:
        """Generate a dynamic observation for the agent."""
        # Select random entities and relations
        num_entities = np.random.randint(2, min(8, len(self.entities)))
        selected_entities = np.random.choice(self.entities, num_entities, replace=False)
        
        # Generate relationships with some causal structure
        relationships = []
        num_relations = np.random.randint(1, max(2, num_entities))
        
        for _ in range(num_relations):
            source = np.random.choice(selected_entities)
            target = np.random.choice([e for e in selected_entities if e != source])
            relation = np.random.choice(self.relations)
            
            # Add some noise and uncertainty
            confidence = np.random.beta(2, 1)  # Skewed towards higher confidence
            
            relationships.append({
                'source': source,
                'target': target,
                'relation': relation,
                'confidence': confidence,
                'timestamp': self.step_count
            })
        
        # Generate state vector with environmental factors
        state_dim = self.config.world_model.state_dim
        state = np.random.normal(0, self.environment_complexity, state_dim)
        
        # Add some structured patterns (hidden causal relationships)
        if 'gravity' in selected_entities and 'sphere_red' in selected_entities:
            state[0] += -9.81  # Gravitational effect
        
        if 'temperature' in selected_entities and 'energy' in selected_entities:
            state[1] += np.random.exponential(2.0)  # Temperature-energy coupling
        
        observation = {
            'entities': list(selected_entities),
            'relations': [(r['source'], r['relation'], r['target']) for r in relationships],
            'state': state,
            'context': f'simulation_step_{self.step_count}',
            'timestamp': self.step_count,
            'environment_complexity': self.environment_complexity,
            'metadata': {
                'session_time': (datetime.now() - self.session_start).total_seconds(),
                'relationship_details': relationships
            }
        }
        
        return observation
    
    def update_environment_complexity(self):
        """Gradually increase environment complexity over time."""
        self.environment_complexity += self.complexity_growth_rate
        
        # Periodically add new entities and relations
        if self.step_count % 1000 == 0:
            new_entity = f'dynamic_entity_{self.step_count}'
            self.entities.append(new_entity)
            
            new_relation = f'emergent_relation_{self.step_count // 1000}'
            self.relations.append(new_relation)
            
            logger.info(f"Environment complexity increased to {self.environment_complexity:.3f}")
            logger.info(f"Added new entity: {new_entity}, new relation: {new_relation}")
    
    def evaluate_agent_performance(self, results: Dict[str, Any]):
        """Evaluate and track agent performance metrics."""
        # Update performance tracking
        self.performance_metrics['total_experiments'] += results.get('experiments_conducted', 0)
        
        # Track prediction accuracy (simplified)
        prediction_error = results.get('prediction_error', 0.5)
        if prediction_error < 0.3:
            self.performance_metrics['successful_predictions'] += 1
        else:
            self.performance_metrics['failed_predictions'] += 1
        
        # Track trends
        self.performance_metrics['confidence_trend'].append(results.get('global_confidence', 0.0))
        self.performance_metrics['knowledge_graph_size'].append(
            results.get('knowledge_graph_stats', {}).get('num_nodes', 0)
        )
        self.performance_metrics['curiosity_rewards'].append(
            results.get('curiosity_reward', 0.0)
        )
        
        # Keep only recent history to prevent memory issues
        max_history = 10000
        for key in ['confidence_trend', 'knowledge_graph_size', 'curiosity_rewards']:
            if len(self.performance_metrics[key]) > max_history:
                self.performance_metrics[key] = self.performance_metrics[key][-max_history:]
    
    def save_progress(self):
        """Save agent state and learning progress."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save agent state
        agent_state = self.agent.get_agent_state()
        agent_file = self.save_dir / f"agent_state_{timestamp}.json"
        
        with open(agent_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_state = self._make_serializable(agent_state)
            json.dump(serializable_state, f, indent=2)
        
        # Save performance metrics
        metrics_file = self.save_dir / f"performance_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            serializable_metrics = self._make_serializable(self.performance_metrics)
            json.dump(serializable_metrics, f, indent=2)
        
        # Save learning history
        history_file = self.save_dir / f"learning_history_{timestamp}.json"
        with open(history_file, 'w') as f:
            json.dump(self.learning_history[-1000:], f, indent=2)  # Last 1000 steps
        
        logger.info(f"Progress saved at step {self.step_count}")
        self.last_save = datetime.now()
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def run_continuous_learning(self, max_steps: Optional[int] = None, 
                              max_runtime: Optional[timedelta] = None):
        """
        Run the continuous learning simulation.
        
        Args:
            max_steps: Maximum number of learning steps (None for infinite)
            max_runtime: Maximum runtime duration (None for infinite)
        """
        logger.info("Starting continuous learning simulation...")
        logger.info(f"Max steps: {max_steps}, Max runtime: {max_runtime}")
        
        try:
            while True:
                # Check termination conditions
                if max_steps and self.step_count >= max_steps:
                    logger.info(f"Reached maximum steps: {max_steps}")
                    break
                
                if max_runtime and (datetime.now() - self.session_start) >= max_runtime:
                    logger.info(f"Reached maximum runtime: {max_runtime}")
                    break
                
                # Generate observation
                observation = self.generate_observation()
                
                # Agent learning step
                start_time = time.time()
                results = self.agent.active_learning_step(observation)
                step_time = time.time() - start_time
                
                # Track learning
                learning_record = {
                    'step': self.step_count,
                    'timestamp': datetime.now().isoformat(),
                    'observation_entities': len(observation['entities']),
                    'observation_relations': len(observation['relations']),
                    'results': results,
                    'step_time': step_time,
                    'environment_complexity': self.environment_complexity
                }
                self.learning_history.append(learning_record)
                
                # Evaluate performance
                self.evaluate_agent_performance(results)
                
                # Update environment
                self.update_environment_complexity()
                
                # Periodic logging
                if self.step_count % 100 == 0:
                    avg_confidence = np.mean(self.performance_metrics['confidence_trend'][-100:])
                    logger.info(
                        f"Step {self.step_count}: "
                        f"Avg confidence: {avg_confidence:.3f}, "
                        f"Experiments: {results.get('experiments_conducted', 0)}, "
                        f"Graph nodes: {results.get('knowledge_graph_stats', {}).get('num_nodes', 0)}, "
                        f"Step time: {step_time:.3f}s"
                    )
                
                # Periodic saving
                if datetime.now() - self.last_save >= self.save_interval:
                    self.save_progress()
                
                self.step_count += 1
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Continuous learning interrupted by user")
        except Exception as e:
            logger.error(f"Error in continuous learning: {e}", exc_info=True)
        finally:
            # Final save
            self.save_progress()
            logger.info(f"Continuous learning completed after {self.step_count} steps")
            
            # Print final statistics
            self._print_final_statistics()
    
    def _print_final_statistics(self):
        """Print comprehensive final statistics."""
        runtime = datetime.now() - self.session_start
        
        print("\n" + "="*80)
        print("CONTINUOUS LEARNING SESSION SUMMARY")
        print("="*80)
        print(f"Total Runtime: {runtime}")
        print(f"Total Steps: {self.step_count}")
        print(f"Steps per second: {self.step_count / runtime.total_seconds():.2f}")
        print(f"Final Environment Complexity: {self.environment_complexity:.3f}")
        print()
        
        print("PERFORMANCE METRICS:")
        print(f"Total Experiments: {self.performance_metrics['total_experiments']}")
        print(f"Successful Predictions: {self.performance_metrics['successful_predictions']}")
        print(f"Failed Predictions: {self.performance_metrics['failed_predictions']}")
        
        if self.performance_metrics['confidence_trend']:
            print(f"Final Confidence: {self.performance_metrics['confidence_trend'][-1]:.3f}")
            print(f"Average Confidence: {np.mean(self.performance_metrics['confidence_trend']):.3f}")
        
        if self.performance_metrics['knowledge_graph_size']:
            print(f"Final Graph Size: {self.performance_metrics['knowledge_graph_size'][-1]} nodes")
            print(f"Graph Growth: {self.performance_metrics['knowledge_graph_size'][-1] - self.performance_metrics['knowledge_graph_size'][0]} nodes")
        
        print(f"Total Entities: {len(self.entities)}")
        print(f"Total Relations: {len(self.relations)}")
        print("="*80)


def main():
    """Main entry point for continuous learning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Learning ICA Agent")
    parser.add_argument("--max-steps", type=int, help="Maximum number of learning steps")
    parser.add_argument("--max-hours", type=float, help="Maximum runtime in hours")
    parser.add_argument("--save-dir", type=str, default="continuous_learning_data",
                       help="Directory to save learning data")
    parser.add_argument("--config", type=str, help="Path to custom configuration file")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    if args.config:
        config = Config.load_from_file(args.config)
    
    # Calculate max runtime
    max_runtime = None
    if args.max_hours:
        max_runtime = timedelta(hours=args.max_hours)
    
    # Create continuous learning environment
    env = ContinuousLearningEnvironment(config, args.save_dir)
    
    # Run continuous learning
    env.run_continuous_learning(
        max_steps=args.max_steps,
        max_runtime=max_runtime
    )


if __name__ == "__main__":
    main()
