# Continuous Learning Setup Guide

This guide explains how to set up and run the ICA agent in continuous learning mode, where it persistently learns from its environment simulation.

## 🚀 Quick Start

### 1. Basic Continuous Learning
```bash
# Simple continuous learning (will run until stopped)
python continuous_learning.py

# Run for specific duration
python continuous_learning.py --max-hours 2

# Run for specific number of steps
python continuous_learning.py --max-steps 10000
```

### 2. Using the Launcher (Recommended)
```bash
# Interactive launcher with preset options
python launch_continuous_learning.py
```

### 3. Real-time Monitoring
```bash
# In a separate terminal, start the monitor
python monitor_continuous_learning.py

# Text-only monitoring (if matplotlib issues)
python monitor_continuous_learning.py --text-only

# Generate HTML dashboard
python monitor_continuous_learning.py --generate-dashboard
```

## 📊 What Happens During Continuous Learning

### Learning Process
1. **Environment Generation**: Creates dynamic observations with entities, relations, and state vectors
2. **Agent Processing**: ICA agent processes observations through its learning pipeline
3. **Knowledge Building**: Builds and refines causal knowledge graph over time
4. **Adaptation**: Adjusts exploration strategies based on accumulated knowledge
5. **Abstraction**: Forms higher-level concepts and motifs from learned patterns

### Environment Dynamics
- **Complexity Growth**: Environment gradually becomes more complex
- **Entity Evolution**: New entities and relations are introduced over time
- **Causal Structure**: Hidden causal relationships for the agent to discover
- **Noise and Uncertainty**: Realistic uncertainty in observations

## 🛠️ Configuration

### Configuration File
Use the optimized configuration for continuous learning:
```bash
python continuous_learning.py --config config/continuous_learning.yaml
```

### Key Settings
```yaml
# Control learning duration
continuous_learning:
  complexity_growth_rate: 0.0005      # How fast environment evolves
  new_entity_interval: 2000           # Add entities every N steps
  checkpoint_interval: 10000          # Save progress every N steps

# Memory management
hardware:
  memory_limit_gb: 8                  # Prevent memory overflow
  garbage_collection_interval: 1000   # Clean up memory
```

## 📈 Monitoring and Analysis

### Real-time Dashboard
The monitoring dashboard shows:
- **Confidence Trend**: Agent's confidence in its knowledge over time
- **Knowledge Graph Growth**: Number of nodes/edges in causal graph
- **Learning Rate**: Number of experiments conducted per step
- **Environment Complexity**: Current complexity of the simulation

### Saved Data
Progress is automatically saved every 10 minutes to:
```
continuous_learning_data/
├── agent_state_YYYYMMDD_HHMMSS.json      # Complete agent state
├── performance_metrics_YYYYMMDD_HHMMSS.json  # Performance metrics
└── learning_history_YYYYMMDD_HHMMSS.json     # Recent learning history
```

### Performance Metrics
- **Total Experiments**: Cumulative experiments conducted
- **Prediction Accuracy**: Success rate of agent's predictions
- **Knowledge Growth**: Evolution of knowledge graph size
- **Curiosity Rewards**: Intrinsic motivation signals

## 🔬 Advanced Usage

### Custom Environment
Create your own environment by modifying the `ContinuousLearningEnvironment` class:

```python
class CustomEnvironment(ContinuousLearningEnvironment):
    def generate_observation(self):
        # Your custom observation generation
        return custom_observation
```

### Multiple Agents
Run multiple agents simultaneously:
```bash
# Agent 1
python continuous_learning.py --save-dir agent1_data &

# Agent 2  
python continuous_learning.py --save-dir agent2_data &

# Monitor both
python monitor_continuous_learning.py --data-dir agent1_data &
python monitor_continuous_learning.py --data-dir agent2_data
```

### Distributed Learning (Future)
```bash
# Master node
python continuous_learning.py --distributed --role master

# Worker nodes
python continuous_learning.py --distributed --role worker --master-ip 192.168.1.100
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Usage**
   - Reduce `max_history_length` in config
   - Lower `buffer_size` for action planner
   - Enable `garbage_collection_interval`

2. **Performance Issues**
   - Use smaller `embedding_dim` and `hidden_dim`
   - Reduce `epistemic_samples` for faster inference
   - Enable CUDA if available

3. **Monitoring Issues**
   - Use `--text-only` flag for monitor
   - Check `data_dir` exists and has write permissions
   - Install matplotlib/plotly for visualizations

### Error Recovery
The system includes automatic error recovery:
- Saves progress every 10 minutes
- Handles NaN values and extreme predictions
- Emergency stops after consecutive errors

## 📋 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB disk space
- CPU with 2+ cores

### Recommended Requirements
- Python 3.9+
- 16GB RAM
- 10GB disk space
- NVIDIA GPU with CUDA support
- SSD for faster I/O

### Dependencies
```bash
# Core dependencies
pip install -r requirements-minimal.txt

# Enhanced features
pip install -r requirements-optional.txt

# For monitoring dashboard
pip install matplotlib plotly
```

## 🎯 Learning Objectives

### Short-term (1-24 hours)
- Basic causal relationship discovery
- Stable knowledge graph construction
- Consistent exploration strategies
- Baseline performance establishment

### Medium-term (1-7 days)
- Complex motif discovery
- Hierarchical concept formation
- Transfer learning between contexts
- Performance optimization

### Long-term (1+ weeks)
- Meta-learning capabilities
- Emergent behavior patterns
- Domain-specific expertise
- Advanced abstraction levels

## 🚦 Safety Considerations

### Automatic Safeguards
- Memory usage monitoring
- Gradient clipping for stability
- Confidence bounds enforcement
- Emergency stop mechanisms

### Manual Monitoring
- Check log files regularly
- Monitor system resources
- Verify checkpoint integrity
- Review performance trends

## 🔮 Future Enhancements

### Planned Features
- Multi-modal learning (vision + language)
- Distributed multi-agent learning
- Real-world environment integration
- Advanced transfer learning
- Automated hyperparameter tuning

### Research Directions
- Causal discovery validation
- Emergent behavior analysis
- Meta-learning evaluation
- Safety and alignment research

---

## 📞 Support

For issues with continuous learning:
1. Check the troubleshooting section above
2. Review log files in `continuous_learning.log`
3. Open an issue on GitHub with:
   - System specifications
   - Configuration used
   - Error messages
   - Performance metrics

**Happy Learning! 🧠✨**
