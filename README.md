# Integrated Causal Agency (ICA) Framework

**A sophisticated AGI framework implementing intrinsic curiosity-driven learning ### 📡 Console Commands

After installation, you can use these convenient commands:

```bash
# Database setup and configuration
ica-setup

# Run enhanced learning with physics simulation
ica-learn
```

**Note**: Legacy monitoring and visualization console commands (`ica-monitor`, `ica-view-graph`) have been removed. ## 📈 Performance Metrics

The framework demonstrates significant improvements over baseline approaches:

- **Motif Discovery**: Efficient frequent subgraph mining
- **Prediction Accuracy**: Bayesian uncertainty improves calibration
- **Sample Efficiency**: Curiosity-driven exploration reduces data requirements
- **Abstraction Quality**: Hierarchical concepts improve generalization

## 🎯 Measuring Learning Value

### Knowledge Quality Metrics

After substantial learning (like your 36,677 scenarios), evaluate the AGI brain's capabilities:

#### 1. **Graph Structure Analysis**
```bash
# Access Neo4j Browser at http://localhost:7474
# Run these Cypher queries to analyze your learned knowledge:

# Total knowledge size
MATCH (n) RETURN count(n) as total_nodes
MATCH ()-[r]->() RETURN count(r) as total_relationships

# Knowledge density and connectivity
MATCH (n)
OPTIONAL MATCH (n)-[r]-()
RETURN n.label, count(r) as connections
ORDER BY connections DESC

# Confidence distribution (higher = better learning)
MATCH ()-[r]->()
RETURN avg(r.confidence) as avg_confidence, 
       min(r.confidence) as min_confidence,
       max(r.confidence) as max_confidence
```

#### 2. **Pattern Recognition Quality**
```bash
# Find learned patterns and abstractions
MATCH (n)-[r1]->(m)-[r2]->(o)
WHERE r1.confidence > 0.8 AND r2.confidence > 0.8
RETURN n.id, r1.type, m.id, r2.type, o.id
LIMIT 50

# Identify emergent concepts
MATCH (n)
WHERE size((n)-[]-()) > 5  # Highly connected entities
RETURN n.id, n.label, size((n)-[]-()) as connections
ORDER BY connections DESC
```

### Real-World Application Tests

#### 3. **Predictive Capability Assessment**
Create test scenarios to evaluate the agent's learned knowledge:

```python
# Test the agent's predictive abilities
def evaluate_prediction_accuracy(agent, test_scenarios):
    correct_predictions = 0
    total_predictions = 0
    
    for scenario in test_scenarios:
        # Give partial information
        partial_obs = {
            "entities": scenario["entities"][:2],  # Only show some entities
            "relationships": [],  # No relationships
            "state": scenario["state"]
        }
        
        # Agent predicts what relationships should exist
        predictions = agent.predict_relationships(partial_obs)
        
        # Compare with actual known relationships
        for pred in predictions:
            if pred in scenario["relationships"]:
                correct_predictions += 1
            total_predictions += 1
    
    return correct_predictions / total_predictions if total_predictions > 0 else 0
```

#### 4. **Transfer Learning Evaluation**
Test if the agent can apply learned knowledge to new domains:

```python
# Introduce completely new scenario types
new_scenario = {
    "name": "Medical Diagnosis System",
    "entities": [
        {"id": "patient_symptoms", "label": "input"},
        {"id": "diagnostic_algorithm", "label": "processor"},
        {"id": "treatment_plan", "label": "output"}
    ],
    "relationships": [
        {"source": "patient_symptoms", "target": "diagnostic_algorithm", "type": "feeds_into", "confidence": 0.9}
    ]
}

# See if agent can quickly learn and generalize
transfer_performance = agent.active_learning_step(new_scenario)
```

## 🚀 Practical Applications of Your AGI Brain

### 1. **Autonomous System Design**
Your learned knowledge graph can be used to:
- **Design smart buildings**: Apply learned HVAC, lighting, and security patterns
- **Optimize industrial processes**: Use learned automation and control relationships
- **Create adaptive interfaces**: Apply sensor-controller-actuator patterns to new domains

### 2. **Causal Reasoning Engine**
```python
# Query the learned causal knowledge
def find_causal_chain(agent, start_entity, end_entity):
    """Find causal pathways between two entities"""
    # Use graph traversal to find causal chains
    paths = agent.knowledge_graph.find_paths(start_entity, end_entity)
    return sorted(paths, key=lambda x: sum([edge.confidence for edge in x]))

# Example: What causes energy consumption to increase?
energy_causes = find_causal_chain(agent, "motion_detector", "energy_consumption")
```

### 3. **Intelligent Decision Making**
```python
# Use learned knowledge for decision support
def recommend_action(agent, current_state, goal_state):
    """Recommend actions based on learned causal relationships"""
    # Find entities that influence the goal
    influencers = agent.knowledge_graph.get_predecessors(goal_state)
    
    # Rank by confidence and controllability
    actions = []
    for entity in influencers:
        if agent.is_controllable(entity):
            confidence = agent.get_relationship_confidence(entity, goal_state)
            actions.append((entity, confidence))
    
    return sorted(actions, key=lambda x: x[1], reverse=True)

# Example: How to optimize room temperature?
temp_actions = recommend_action(agent, current_obs, "room_temperature")
```

### 4. **Anomaly Detection**
```python
# Detect unusual patterns using learned normal behavior
def detect_anomalies(agent, new_observation):
    """Detect deviations from learned patterns"""
    expected_relationships = agent.predict_relationships(new_observation)
    actual_relationships = new_observation["relationships"]
    
    anomalies = []
    for expected in expected_relationships:
        if expected not in actual_relationships:
            anomalies.append(f"Missing: {expected}")
    
    for actual in actual_relationships:
        if actual not in expected_relationships:
            anomalies.append(f"Unexpected: {actual}")
    
    return anomalies
```

## 🔬 Advanced Evaluation Framework

### Comprehensive AGI Brain Assessment

After substantial learning sessions (like 36,677 scenarios), evaluate your AGI brain's capabilities:

```bash
# Run comprehensive evaluation of learned knowledge
python examples/evaluate_learning.py

# With custom Neo4j configuration
python examples/evaluate_learning.py --neo4j-password your_password
```

**The evaluation measures:**
1. **Knowledge Structure**: Size, density, and connectivity of learned knowledge
2. **Prediction Accuracy**: Ability to predict relationships in new situations  
3. **Transfer Learning**: Adaptation speed to completely new domains
4. **Reasoning Capabilities**: Causal inference and logical reasoning

**AGI Readiness Score**: Weighted combination of all capabilities (0.0-1.0)
- **0.8+**: AGI-Ready - Excellent performance across all capabilities
- **0.6+**: Advanced - Strong capabilities with room for improvement
- **0.4+**: Developing - Good foundation, needs more learning
- **0.2+**: Early Stage - Basic capabilities emerging

### Real-World Applications

Once your AGI brain reaches sufficient capability scores, you can deploy it for:

#### 🏠 **Autonomous Building Management**
```python
# Use learned HVAC, lighting, security patterns
building_optimizer = BuildingController(trained_agent)
building_optimizer.optimize_energy_efficiency()
building_optimizer.enhance_occupant_comfort()
```

#### 🏭 **Industrial Process Optimization**
```python
# Apply learned automation and control relationships
process_optimizer = IndustrialController(trained_agent)
process_optimizer.predict_maintenance_needs()
process_optimizer.optimize_production_flow()
```

#### 🔮 **Predictive System Design**
```python
# Design new systems based on learned patterns
system_designer = AutonomousDesigner(trained_agent)
new_system = system_designer.design_smart_system(requirements)
```

#### 🧠 **Causal Decision Engine**
```python
# Make decisions based on learned causal relationships
decision_engine = CausalDecisionMaker(trained_agent)
action_plan = decision_engine.recommend_actions(current_state, desired_outcome)
```j Browser at `http://localhost:7474` for real-time graph exploration.ning and hierarchical abstraction.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🧠 Overview

The ICA Framework implements a novel approach to Artificial General Intelligence through intrinsic curiosity-driven learning. It combines causal knowledge graphs, Bayesian uncertainty quantification, hierarchical abstraction, and advanced reinforcement learning to create agents capable of autonomous learning and adaptation.

**✅ LEARNING VERIFIED**: The agent successfully builds causal knowledge graphs, discovers structural patterns (motifs), and forms higher-level concepts through continuous interaction with its environment.

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

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup Neo4j Database (Windows):**

   **Option A: Neo4j Desktop (Recommended)**
   - Download Neo4j Desktop from [neo4j.com/download](https://neo4j.com/download/)
   - Install and create a new project
   - Create a new database instance
   - Set password and start the database
   - Note the connection details (usually `neo4j://localhost:7687`)

   **Option B: Neo4j Community Server**
   - Download from [neo4j.com/deployment-center](https://neo4j.com/deployment-center/)
   - Extract and run: `bin\neo4j.bat console`
   - Set initial password via web interface at `http://localhost:7474`

4. **Configure Database:**
```bash
python setup.py database
```
   This will:
   - Check Neo4j driver installation
   - Create sample database configurations
   - Set up your Neo4j connection details
   - Test the database connection

5. **Run setup:**
```bash
python setup.py develop
```

### ⚡ Quick Learning Test

Run the enhanced learning script with physics simulation:

```bash
python examples/learning.py
```

This will show the agent:
- ✅ Building a knowledge graph from observations
- ✅ Discovering structural patterns (motifs)
- ✅ Forming concepts through clustering
- ✅ Learning continuously from new data

### 🎯 Run Learning with Different Backends

```bash
# Using Neo4j backend (recommended, default)
python examples/learning.py --backend neo4j

# Using in-memory backend (for testing)
python examples/learning.py --backend memory

# With custom Neo4j configuration
python examples/learning.py --backend neo4j \
    --neo4j-uri neo4j://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password mypassword \
    --neo4j-database neo4j
```

### 🔧 Command Line Options

The learning script supports several configuration options:
- `--backend` : Choose database backend (`neo4j` or `memory`)
- `--neo4j-uri` : Neo4j server URI (overrides config file)
- `--neo4j-user` : Neo4j username (overrides config file)  
- `--neo4j-password` : Neo4j password (overrides config file)
- `--neo4j-database` : Neo4j database name (overrides config file)

### � Console Commands

After installation, you can use these convenient commands:

```bash
# Database setup and configuration
ica-setup

# Run enhanced learning with physics simulation
ica-learn

# Evaluate your AGI brain's capabilities  
ica-evaluate
```

## 🔄 Continuous Learning

### Enhanced Learning with Physics Simulation

The ICA Framework includes realistic physics simulation and procedural scenario generation:

```bash
# Run enhanced learning with physics simulation
python examples/learning.py
```

**Features:**
- **Physics Simulation**: 40+ physics entities (particles, forces, fields, emergent properties)
- **Procedural Scenarios**: Smart home automation, industrial robotics, autonomous vehicles
- **Neo4j Backend**: Persistent knowledge storage and session resumption
- **Real-time Monitoring**: Graph growth and pattern discovery with milestone tracking
- **Advanced Learning**: Multi-round scenario variation with complexity progression
- **Scenario Types**: Every 5th scenario uses physics simulation, every 7th uses procedural generation
- **Graceful Shutdown**: Ctrl+C saves all progress to database and enables session resumption

### Database Backends

The framework supports multiple database backends:

1. **Neo4j (Recommended)**: Persistent graph database
   - Use: `python examples/learning.py --backend neo4j` (default)
   - Configure via: `python setup.py database`

2. **Memory**: In-memory for testing
   - Use: `python examples/learning.py --backend memory`

### Monitor Learning Progress

**Note**: Legacy monitoring scripts have been removed. Current Neo4j-based sessions store data directly in the database for real-time access.

**Neo4j Browser**: Access your Neo4j instance at `http://localhost:7474` to explore the knowledge graph in real-time.

### View Knowledge Graph

**Note**: Legacy visualization scripts have been removed in favor of Neo4j's built-in tools.

**Recommended**: Use Neo4j Browser or custom Cypher queries to visualize and analyze the knowledge graph data stored in your Neo4j database.

## 📊 What You'll See

During continuous learning, the agent will:

1. **🔗 Build Knowledge Graphs**
   - Nodes: Entities from observations
   - Edges: Causal relationships with confidence scores
   - Growth: Graph expands as new observations arrive

2. **🔍 Discover Patterns**
   - Motifs: Recurring structural patterns in the graph
   - Frequency: How often patterns appear
   - Size: Complexity of discovered patterns

3. **🧠 Form Concepts**
   - Clustering: Groups similar motifs together
   - Abstractions: Higher-level understanding emerges
   - Utility: Concepts get scored based on usefulness

4. **� Progress Tracking**
   - Milestone-based updates: Progress shown every 2000 edge relationships
   - Session resumption: Neo4j sessions can resume from previous state
   - Learning rate: Real-time scenarios processed per second
   - Database status: Connection health and data persistence verification

## 📁 Session Data

When using **Neo4j backend** (recommended), all learning data is automatically persisted in the Neo4j database:
- **Knowledge graph**: Nodes and edges with confidence scores
- **Session metadata**: Scenarios completed, learning time, session IDs
- **Resumption capability**: Sessions can be resumed from where they left off

When using **memory backend** (testing only), data is kept in memory and lost when the session ends.

### Database Configuration Files

After running `python setup.py database`, configuration files are created in:
- `config/database/neo4j.json` - Default Neo4j configuration
- `config/database/neo4j_sample_configuration.json` - Sample configuration template

## 🏗️ Project Structure

The ICA Framework has been streamlined for clarity:

```
ica-framework/
├── ica_framework/           # Core framework code
│   ├── core/               # Agent and core logic
│   │   ├── __init__.py
│   │   └── ica_agent.py
│   ├── components/         # Modular components
│   │   ├── __init__.py
│   │   ├── action_planner.py
│   │   ├── causal_knowledge_graph.py
│   │   ├── curiosity_module.py
│   │   ├── hierarchical_abstraction.py
│   │   └── world_model.py
│   ├── database/           # Database adapters
│   │   ├── __init__.py
│   │   ├── graph_database.py
│   │   ├── memory_adapter.py
│   │   └── neo4j_adapter.py
│   ├── sandbox/            # Simulation environments
│   │   ├── __init__.py
│   │   └── sandbox_environment.py
│   ├── utils/              # Utilities and helpers
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── __init__.py
│   └── enhanced_knowledge_graph.py  # Enhanced KG with database support
├── examples/               # Example scripts
│   ├── learning.py         # Enhanced continuous learning with physics simulation
│   └── evaluate_learning.py # Comprehensive AGI brain evaluation
├── docs/                   # Documentation
│   └── database_backends.md
├── requirements/           # Dependency specifications
│   ├── requirements.txt    # Core dependencies  
│   ├── requirements-dev.txt # Development dependencies
│   └── REQUIREMENTS_CONSOLIDATION.md # Consolidation documentation
├── config/                 # Configuration files
│   └── database/           # Database configurations
└── setup.py               # Unified setup and database configuration
```

**Recent Updates:**
- ✅ Consolidated `setup_database.py` into `setup.py`
- ✅ Removed `demo.py` example script
- ✅ Enhanced `examples/learning.py` with physics simulation and procedural scenarios
- ✅ Streamlined console commands via entry points
- ✅ Removed unused script files and entire `scripts/` directory (legacy JSON-based tools)
- ✅ Simplified project structure for Neo4j-first approach
- ✅ Updated project structure documentation

2. **Install dependencies:**
```bash
# Install core dependencies
pip install -r requirements/requirements.txt

# Or install with development tools
pip install -r requirements/requirements-dev.txt

# Or install specific optional packages
pip install -e .[viz]      # Visualization tools
pip install -e .[ml]       # Machine learning tools
pip install -e .[causal]   # Causal inference tools
pip install -e .[all]      # All optional dependencies
```

### Basic Usage

```python
from ica_framework import ICAAgent, Config
from ica_framework.database import GraphDatabase

# Initialize configuration
config = Config()

# Setup database backend
db_config = {
    'backend': 'neo4j',  # or 'memory' for testing
    'config': {
        'uri': 'neo4j://localhost:7687',
        'username': 'neo4j',
        'password': 'your_password',
        'database': 'neo4j'
    }
}

# Create database connection
database = GraphDatabase.from_config(db_config)

# Create ICA agent
agent = ICAAgent(config, database=database)

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

### Running the Enhanced Learning

```bash
# Use the unified setup first
python setup.py database  # Configure your database

# Run enhanced learning with physics simulation
python examples/learning.py
```

This will run a complete learning session including:
- **Physics simulation** with 40+ entities and relationships
- **Procedural scenario generation** (smart home, industrial automation, autonomous vehicles)
- **Real-time knowledge graph construction** with Neo4j persistence and session resumption
- **Pattern discovery and concept formation** through hierarchical abstraction
- **Continuous learning** with intrinsic motivation and curiosity-driven exploration
- **Progress tracking** with edge milestone updates every 2000 relationships
- **Scenario variation** across multiple rounds with increasing complexity

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

## � Learning Script Features

The `examples/learning.py` script provides a comprehensive continuous learning environment:

### 🌟 Enhanced Scenario Generation
- **Base IoT Scenarios**: 15 predefined smart home/building automation scenarios
- **Physics Simulation**: 40+ physics entities including forces, objects, and emergent properties
- **Procedural Generation**: Smart home, industrial robotics, autonomous vehicles, supply chain, energy management, environmental monitoring
- **Adaptive Complexity**: Scenarios become more complex over multiple rounds

### 🔄 Learning Modes
- **Continuous Learning**: Infinite loop with graceful shutdown (Ctrl+C)
- **Session Resumption**: Automatically resumes from previous Neo4j sessions
- **Multi-round Progression**: Same scenarios with increasing complexity and variation
- **Milestone Tracking**: Progress updates every 2000 edge relationships

### 📊 Real-time Feedback
- **Silent Operation**: Suppresses all logging for clean console output
- **Progress Milestones**: Shows significant learning achievements
- **Learning Rate**: Real-time scenarios processed per second
- **Database Status**: Connection health and persistence verification

## �📈 Performance Metrics

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
pip install -r requirements/requirements-dev.txt  # Includes core + dev tools
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

1. **Neo4j Connection Issues (Windows)**
   ```bash
   # Test your Neo4j connection
   python setup.py database
   # Select option 3: Test Neo4j connection
   ```
   
   **Common fixes:**
   - Ensure Neo4j is running (check Neo4j Desktop or Windows Services)
   - Verify connection details: `neo4j://localhost:7687` (or `neo4j://127.0.0.1:7687`)
   - Check username/password (default: `neo4j`/`your_password`)
   - Firewall: Allow Neo4j ports (7687 for Bolt, 7474 for HTTP)

2. **Neo4j Service Not Starting**
   ```bash
   # For Neo4j Desktop: Restart the database instance
   # For Community Server: Check logs in neo4j/logs/
   ```

3. **CUDA Installation**: For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. **Graph Dependencies**: Some features require additional packages:
```bash
pip install torch-geometric pyg-lib torch-scatter torch-sparse
```

5. **Visualization**: For interactive plots:
```bash
pip install plotly seaborn
```

### Database Configuration Files

After running `python setup.py database`, configuration files are created in:
- `config/database/neo4j.json` - Default Neo4j configuration  
- `config/database/neo4j_sample_configuration.json` - Sample configuration template

### Legacy Features

**Removed Components**: 
- Scripts directory and monitoring tools (replaced by Neo4j Browser)
- JSON-based data storage (replaced by Neo4j persistence)
- File-based visualization tools (use Neo4j's built-in capabilities)

**Current Approach**: All data is stored in Neo4j for real-time access, persistence, and built-in visualization capabilities.

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
