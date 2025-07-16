# Core Components Directory

This directory contains the modular core components that replaced the large monolithic `main_runner.py` file.

## Directory Structure

```
core/
├── __init__.py                          # Package initialization
├── agi_runner.py                        # Main system coordinator
├── component_initializer.py             # Component initialization
├── learning_coordinator.py             # Learning process management
├── main_loop_controller.py             # Main execution loop
├── session_manager.py                  # Session restoration
├── shutdown_manager.py                 # Graceful shutdown
└── neural_persistence_safeguards.py    # Documentation
```

## Components Overview

### 1. AGIRunner (agi_runner.py)
- **Purpose**: Main system coordinator
- **Responsibilities**: Orchestrates all other components
- **Key Features**: System initialization, learning coordination, shutdown management

### 2. ComponentInitializer (component_initializer.py)
- **Purpose**: Initialize all system components
- **Responsibilities**: GPU processor, knowledge graph, database manager, simulators
- **Key Features**: Dependency-ordered initialization, error handling

### 3. SessionManager (session_manager.py)
- **Purpose**: Session restoration and persistence
- **Responsibilities**: Check for previous learning data, restore session state
- **Key Features**: TRUE learning persistence, fallback restoration

### 4. LearningCoordinator (learning_coordinator.py)
- **Purpose**: Coordinate the learning process
- **Responsibilities**: Start/stop learning, monitor learning state
- **Key Features**: AGI agent coordination, monitoring integration

### 5. MainLoopController (main_loop_controller.py)
- **Purpose**: Main execution loop control
- **Responsibilities**: System health checks, periodic saves, main loop
- **Key Features**: **Periodic neural network weight saving every 2 minutes**

### 6. ShutdownManager (shutdown_manager.py)
- **Purpose**: Graceful shutdown with complete state preservation
- **Responsibilities**: Save learning state, stop components, cleanup
- **Key Features**: **Emergency neural network save on manual termination**

## Neural Network Weight Protection 🛡️

The system has multiple safeguards to ensure neural network weights and biases are never lost:

1. **Periodic Save**: Every 2 minutes during operation
2. **Manual Termination**: Ctrl+C triggers graceful shutdown with save
3. **Emergency Save**: Fallback save if regular save fails
4. **Signal Handling**: Proper handling of all termination signals

## Usage

### Modern Approach (Recommended)
```python
from components.core import AGIRunner

runner = AGIRunner()
runner.run()
```

### Legacy Approach (Backwards Compatible)
```python
from components.main_runner import TrueAGIRunner

runner = TrueAGIRunner()  # Shows legacy warning
runner.run()
```

## Migration Benefits

- **Maintainability**: Each component has a single responsibility
- **Testability**: Components can be tested independently
- **Extensibility**: Easy to add new features to specific components
- **Reliability**: Better error handling and recovery
- **Performance**: More efficient resource management

## File Size Reduction

- **Before**: main_runner.py (395 lines)
- **After**: 6 focused components (average 120 lines each)
- **Total**: Better organization and maintainability

## Key Features Preserved

- ✅ Complete learning state persistence
- ✅ Session restoration
- ✅ GPU optimization
- ✅ Periodic saving
- ✅ Graceful shutdown
- ✅ **Neural network weight protection**
- ✅ Database integration
- ✅ Monitoring and statistics

The modular approach provides better maintainability while preserving all existing functionality and adding enhanced neural network weight protection.
