# Integrated Causal Agency (ICA) Framework

**A sophisticated AGI framework implementing intrinsic curiosity-driven learning with causal reasoning and hierarchical abstraction.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🧠 Overview

The ICA Framework implements a novel approach to Artificial General Intelligence through intrinsic curiosity-driven learning. It combines causal knowledge graphs, Bayesian uncertainty quantification, hierarchical abstraction, and advanced reinforcement learning to create agents capable of autonomous learning and adaptation.

## 🏗️ Architecture

### Core Components

1. **🕸️ Causal Knowledge Graph**
   - Dynamic graph construction with confidence tracking
   - R-GCN (Relational Graph Convolutional Networks) for representation learning
   - Bayesian confidence updates and edge pruning

2. **🌍 World Model**
   - Bayesian Graph Neural Networks with uncertainty quantification
   - Predictive modeling with aleatoric and epistemic uncertainty
   - CUDA-accelerated training and inference

3. **🔍 Curiosity Module**
   - Intrinsic motivation through prediction error
   - Information-theoretic reward signals
   - Adaptive exploration strategies

4. **🎯 Action Planner**
   - Soft Actor-Critic (SAC) with automatic entropy tuning
   - Hindsight Experience Replay for sample efficiency
   - Multi-objective optimization with curiosity integration

5. **🔄 Hierarchical Abstraction**
   - Motif discovery using frequent subgraph mining
   - Graph embeddings and concept induction
   - Utility-driven abstraction refinement

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/erangross27/Integrated-Casual-Agency.git
cd Integrated-Casual-Agency
```

2. **Install basic dependencies:**
```bash
pip install -r requirements/requirements-minimal.txt
```

3. **For enhanced features (optional):**
```bash
pip install -r requirements/requirements-optional.txt
```

### Basic Usage

```python
from ica_framework import ICAAgent, Config

# Initialize configuration
config = Config()

# Create ICA agent
agent = ICAAgent(config)

# Process observations
for step in range(100):
    observation = {
        'entities': ['entity_1', 'entity_2'],
        'relations': [('entity_1', 'interacts_with', 'entity_2')],
        'context': 'learning_environment',
        'timestamp': step
    }
    
    # Agent learning step
    results = agent.active_learning_step(observation)
    
    print(f"Step {step}: {results['experiments_conducted']} experiments, "
          f"confidence: {results['global_confidence']:.3f}")

# Get final state
final_state = agent.get_agent_state()
print(f"Final: {final_state['knowledge_graph']['num_nodes']} nodes, "
      f"confidence: {final_state['global_confidence']:.3f}")
```

### Running the Demo

```bash
# Use the unified launcher (recommended)
python ica_launcher.py

# Or run directly
python examples/demo.py
```

This will run a complete demonstration including:
- Ablation studies comparing baseline vs enhanced learning
- Individual agent usage examples
- Results visualization and analysis
- Model checkpointing and persistence

## 📊 Features

### ✅ Implemented
- **Causal Reasoning**: Dynamic knowledge graph construction with confidence tracking
- **Uncertainty Quantification**: Bayesian neural networks for epistemic and aleatoric uncertainty
- **Intrinsic Motivation**: Curiosity-driven exploration through prediction error
- **Hierarchical Learning**: Multi-level abstraction from motifs to concepts
- **Advanced RL**: SAC with automatic entropy tuning and experience replay
- **Robust Dependencies**: Graceful fallbacks for optional packages
- **CUDA Support**: GPU acceleration where available
- **Comprehensive Logging**: Structured logging with multiple backends
- **Visualization**: Interactive and static plotting with multiple backends

### 🔬 Research Features
- **Motif Discovery**: Frequent subgraph mining for pattern recognition
- **Concept Induction**: Clustering-based concept formation
- **Utility Optimization**: Dynamic abstraction refinement
- **Meta-Learning**: Transfer of learned abstractions
- **Ablation Framework**: Built-in experimental validation

## 🛠️ Configuration

The framework uses hierarchical configuration through YAML files:

```yaml
# config/default.yaml
causal_graph:
  initial_nodes: 500
  initial_edges: 1000
  confidence_threshold: 0.1
  max_nodes: 50000

world_model:
  hidden_dim: 256
  num_layers: 3
  dropout: 0.1
  learning_rate: 0.001

curiosity:
  prediction_weight: 1.0
  entropy_weight: 0.1
  eta: 0.01

action_planning:
  actor_lr: 0.0003
  critic_lr: 0.0003
  buffer_size: 100000
  batch_size: 256
  gamma: 0.99
  tau: 0.005

abstraction:
  motif_min_size: 2
  motif_max_size: 5
  clustering_algorithm: "kmeans"
  num_clusters: 10
  utility_decay: 0.95
```

## 📈 Performance

The framework demonstrates significant improvements over baseline approaches:

- **Motif Discovery**: Efficient frequent subgraph mining
- **Prediction Accuracy**: Bayesian uncertainty improves calibration
- **Sample Efficiency**: Curiosity-driven exploration reduces data requirements
- **Abstraction Quality**: Hierarchical concepts improve generalization

## 🔬 Research Applications

### Supported Domains
- **Causal Discovery**: Learning causal relationships from observational data
- **Graph Neural Networks**: Advanced GNN architectures with uncertainty
- **Reinforcement Learning**: Curiosity-driven and intrinsically motivated agents
- **Meta-Learning**: Transfer learning through hierarchical abstractions
- **Cognitive Modeling**: Human-like learning and reasoning patterns

### Experimental Framework
- **Ablation Studies**: Built-in experimental validation
- **Metrics Tracking**: Comprehensive performance monitoring
- **Visualization**: Real-time and post-hoc analysis tools
- **Reproducibility**: Deterministic seeding and configuration management

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. **Fork and clone the repository**
2. **Install development dependencies:**
```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt  # For testing and linting
```
3. **Run tests:**
```bash
python -m pytest tests/
```

## 📚 Documentation

- **[API Reference](docs/api.md)**: Detailed API documentation
- **[Architecture Guide](docs/architecture.md)**: In-depth system design
- **[Examples](examples/)**: Code examples and tutorials
- **[Research Papers](docs/papers.md)**: Related publications and theory

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Installation**: For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. **Graph Dependencies**: Some features require additional packages:
```bash
pip install torch-geometric pyg-lib torch-scatter torch-sparse
```

3. **Visualization**: For interactive plots:
```bash
pip install plotly seaborn
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Author**: Eran Gross
- **Email**: erangross27@gmail.com
- **GitHub**: [@erangross27](https://github.com/erangross27)

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- NetworkX for graph algorithms
- scikit-learn for machine learning utilities
- The open-source community for inspiration and tools

## 📊 Citation

If you use this framework in your research, please cite:

```bibtex
@software{gross2025ica,
  title={Integrated Causal Agency: A Framework for Intrinsic Curiosity-Driven Learning},
  author={Gross, Eran},
  year={2025},
  url={https://github.com/erangross27/Integrated-Casual-Agency}
}
```

---

**Built with ❤️ for the AGI research community**
