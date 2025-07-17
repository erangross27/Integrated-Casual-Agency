# Database Components - PostgreSQL-Only TRUE AGI Storage

This folder contains the PostgreSQL-only database components for the TRUE AGI system. **Neural networks ARE the knowledge** - no graph database needed.

## 🏗️ Architecture Philosophy

The TRUE AGI system uses a **single PostgreSQL database** to store:
- **Neural Network Weights**: The actual learned knowledge (pattern recognizer, hypothesis generator)
- **Learning Events**: Environmental interactions and outcomes
- **Pattern Recognition Results**: Recognized patterns with confidence scores
- **Hypothesis Generation Results**: Generated hypotheses and test results
- **Learning Metrics**: Training progress and performance data

## 📁 Core Components

### `postgresql_agi_persistence.py`
- **Primary storage engine** for TRUE AGI learning
- Handles multi-GB neural network weight storage with compression
- Manages learning events, pattern recognition, and hypothesis generation
- Creates and manages PostgreSQL tables optimized for AGI learning

### `neural_persistence.py`
- **Neural network persistence layer** 
- Provides clean interface for saving/loading PyTorch models
- Handles model versioning and cleanup
- Integrates with PostgreSQL AGI persistence

### `database_manager.py`
- **PostgreSQL-only database manager**
- Orchestrates all database operations
- Provides unified interface for storing/restoring complete learning state
- Handles legacy compatibility methods

### `__init__.py`
- Python package initialization
- Exports main database components

## 🐘 PostgreSQL Tables

### Core Tables
- **`agi_sessions`**: Learning sessions and metadata
- **`neural_models`**: Neural network model metadata and versions
- **`neural_weights`**: Compressed binary neural network weights (BYTEA)
- **`learning_events`**: AGI environmental interactions
- **`pattern_recognitions`**: Pattern recognition results
- **`hypothesis_generations`**: Hypothesis generation results
- **`learning_metrics`**: Learning progress metrics

### Key Features
- **Binary Storage**: Efficient BYTEA columns for multi-GB neural weights
- **Compression**: Gzip compression for optimal storage (typically 2-3x reduction)
- **Versioning**: Multiple model versions with current/historical tracking
- **Integrity**: SHA256 checksums for data integrity verification
- **Performance**: Optimized indexes for fast querying

## 🚀 Usage

### Basic Usage
```python
from database_manager import DatabaseManager

# Initialize PostgreSQL-only database
db_manager = DatabaseManager()

# Store complete learning state
db_manager.store_learning_state(agi_agent, gpu_processor)

# Restore learning state
db_manager.restore_learning_state(agi_agent, gpu_processor)

# Log learning events
db_manager.log_learning_event('exploration', environment_state, agi_action)
```

### Direct Neural Persistence
```python
from neural_persistence import NeuralPersistence

# Initialize neural persistence
neural_persistence = NeuralPersistence(session_id)

# Save neural network model
neural_persistence.save_model_weights('pattern_recognizer', model)

# Load neural network model
neural_persistence.load_model_weights('pattern_recognizer', model)
```

## 🔧 Configuration

Database configuration is stored in `../../config/database/database_config.json`:

```json
{
    "database": {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "ica_neural",
        "user": "ica_user",
        "password": "ica_password"
    }
}
```

## 🛠️ Setup

1. **Install PostgreSQL** (Windows installer recommended)
2. **Run setup script**: `python scripts/setup_databases.py`
3. **Verify connection**: Script will test database connectivity

## ✅ Advantages of PostgreSQL-Only Architecture

1. **Simplicity**: Single database to maintain and configure
2. **Efficiency**: Native binary storage for multi-GB neural networks
3. **Reliability**: ACID compliance and proven enterprise stability
4. **Performance**: Optimized for large binary data storage
5. **No Complexity**: No Neo4j installation or maintenance needed
6. **Focus**: Neural networks store implicit knowledge relationships

## 🧠 TRUE AGI Learning Storage

The PostgreSQL database efficiently stores:
- **4GB+ neural network weights** (compressed to ~1-2GB)
- **Environmental learning events** with full context
- **Pattern recognition results** with confidence scores
- **Hypothesis generation** with validation outcomes
- **Learning progression metrics** for monitoring

The system recognizes that **neural networks themselves contain the learned knowledge** through their weights and biases, eliminating the need for separate graph databases.

---

*This PostgreSQL-only architecture provides a clean, efficient foundation for TRUE AGI environmental learning.*
