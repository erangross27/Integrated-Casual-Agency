"""
Example script demonstrating ICA Framework usage
"""

import numpy as np
import networkx as nx
from pathlib import Path
import json

# Add the parent directory to the path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from ica_framework import ICAAgent, Config
from ica_framework.sandbox import SandboxEnvironment


def main():
    """Main demonstration function"""
    
    print("ICA Framework Demonstration")
    print("=" * 40)
    
    # 1. Create configuration
    config = Config()
    config.sandbox.dataset_size = 100  # Smaller for demo
    config.sandbox.random_seed = 42
    
    print(f"Configuration loaded:")
    print(f"  - Dataset size: {config.sandbox.dataset_size}")
    print(f"  - Random seed: {config.sandbox.random_seed}")
    print()
    
    # 2. Set up sandbox environment
    print("Setting up sandbox environment...")
    sandbox = SandboxEnvironment(config.sandbox)
    
    # 3. Run ablation study
    print("Running ablation study...")
    results = sandbox.run_ablation_study()
    
    # 4. Generate report
    print("\nGenerating report...")
    report = sandbox.generate_report(results)
    print(report)
    
    # 5. Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "ablation_study_results.json"
    sandbox.save_results(results, str(results_file))
    
    report_file = results_dir / "ablation_study_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    
    # 6. Demonstrate individual agent usage
    print("\nDemonstrating individual agent usage...")
    
    # Create agent
    agent = ICAAgent(config)
    
    # Initialize with mock dimensions
    agent.initialize_world_model(
        state_dim=10,
        action_dim=4,
        num_relations=5
    )
    
    # Create mock observations
    observations = []
    for i in range(5):
        obs = {
            "entities": [
                {"id": f"entity_{i}", "label": f"type_{i%3}"}
            ],
            "relationships": [
                {
                    "source": f"entity_{i}",
                    "target": f"entity_{(i+1)%5}",
                    "type": "connects_to",
                    "confidence": 0.8
                }
            ],
            "state": np.random.normal(0, 0.1, 10)
        }
        observations.append(obs)
    
    # Run agent steps
    for i, obs in enumerate(observations):
        print(f"  Step {i+1}: Processing observation...")
        step_results = agent.active_learning_step(obs)
        print(f"    - Experiments conducted: {step_results['experiments_conducted']}")
        print(f"    - Global confidence: {step_results['global_confidence']:.3f}")
        print(f"    - Intrinsic reward: {step_results['intrinsic_reward']:.3f}")
    
    # Display final agent state
    final_state = agent.get_agent_state()
    print(f"\nFinal agent state:")
    print(f"  - Total steps: {final_state['step_count']}")
    print(f"  - Graph nodes: {final_state['graph_stats']['num_nodes']}")
    print(f"  - Graph edges: {final_state['graph_stats']['num_edges']}")
    print(f"  - Global confidence: {final_state['global_confidence']:.3f}")
    
    # Save agent
    agent_dir = Path("saved_agents") / "demo_agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    
    if agent.save_agent(str(agent_dir)):
        print(f"\nAgent saved to: {agent_dir}")
    
    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
