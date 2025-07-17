# Database Components - Modern Hybrid Storage for TRUE AGI

This folder contains the modern hybrid database components for the TRUE AGI system. **Neural networks ARE the knowledge** - stored efficiently on disk with PostgreSQL for events.

## 🏗️ Modern Architecture Philosophy

The TRUE AGI system uses a **hybrid storage approach**:
- **🧠 Neural Network Weights**: Stored as PyTorch `.pth` files on disk (industry standard)
- **📊 Learning Events**: PostgreSQL for environmental interactions and outcomes  
- **📋 Metadata**: JSON files with model info, checksums, and timestamps
- **🗄️ Large Model Backup**: HDF5 format for models >1GB with efficient compression

## 📁 Core Components

### `modern_neural_persistence.py` ⭐ **NEW**
- **Modern file-based neural storage** using PyTorch native format
- Handles multi-GB models without memory allocation issues
- Automatic HDF5 backup for very large models (>1GB)
- File integrity checking with SHA256 checksums
- Storage usage monitoring and cleanup

### `modern_database_manager.py` ⭐ **NEW**
- **Hybrid database manager** combining file storage + PostgreSQL
- Neural networks → Disk files (fast, reliable, industry standard)
- Learning events → PostgreSQL (structured, queryable)
- Automatic fallback if PostgreSQL unavailable

### `postgresql_agi_persistence.py`
- **PostgreSQL storage engine** for learning events and metadata
- Handles learning events, pattern recognition results
- Session management and progress tracking
- Compatible with file-based neural storage

### Legacy Components (Maintained for Compatibility)
- `neural_persistence.py`: Legacy PostgreSQL neural storage
- `database_manager.py`: Legacy PostgreSQL-only manager

## 🗂️ Storage Structure

### File-based Neural Storage
```
./agi_checkpoints/
├── session_12345/
│   ├── models/
│   │   ├── pattern_recognizer_latest.pth      # 3.2GB PyTorch model
│   │   ├── hypothesis_generator_latest.pth    # 2.8GB PyTorch model
│   │   └── pattern_recognizer_backup.h5       # HDF5 backup (compressed)
│   └── metadata/
│       ├── pattern_recognizer_info.json       # Model metadata & checksum
│       └── hypothesis_generator_info.json     # Model metadata & checksum
```
### PostgreSQL Tables (Learning Events & Metadata)
- **`agi_sessions`**: Learning sessions and metadata
- **`learning_events`**: AGI environmental interactions
- **`pattern_recognitions`**: Pattern recognition results  
- **`hypothesis_generations`**: Hypothesis generation results
- **`learning_metrics`**: Learning progress metrics

## 🚀 Usage

### Modern Hybrid Storage
```python
from modern_database_manager import create_modern_database_manager

# Initialize modern hybrid database
db_manager = create_modern_database_manager("my_session")

# Store complete learning state (files + PostgreSQL)
db_manager.store_learning_state(agi_agent, gpu_processor)

# Restore learning state (files + PostgreSQL)
db_manager.restore_learning_state(agi_agent, gpu_processor)

# Get storage information
db_manager.get_storage_info()
```

### Direct File-based Neural Storage
```python
from modern_neural_persistence import ModernNeuralPersistence

# Initialize modern neural persistence
neural_storage = ModernNeuralPersistence("my_session")

# Save neural network model (no memory issues!)
neural_storage.save_neural_model('pattern_recognizer', model)

# Load neural network model
neural_storage.load_neural_model('pattern_recognizer', model)

# List all saved models
neural_storage.list_saved_models()
```

## 🔧 Configuration

Neural networks are stored in `./agi_checkpoints/` by default.
PostgreSQL configuration (optional) in `../../config/database/database_config.json`:

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

1. **No special setup required** - files stored locally
2. **Optional PostgreSQL**: For learning event storage
3. **Install HDF5**: `pip install h5py` (for large model backups)

## ✅ Advantages of Modern Hybrid Architecture

1. **🚀 Performance**: No memory allocation issues with large models
2. **📁 Industry Standard**: PyTorch `.pth` format (same as OpenAI, Google)
3. **⚡ Speed**: Faster saves/loads with native PyTorch I/O
4. **🔍 Integrity**: SHA256 checksums and metadata tracking
5. **📊 Analytics**: PostgreSQL for learning event queries (optional)
6. **☁️ Scalable**: Files can be moved to cloud storage easily
7. **🧠 Separation**: Neural weights separate from event data

## 🧠 TRUE AGI Learning Storage

The modern hybrid system efficiently handles:
- **🧠 Multi-GB neural networks** (pattern recognizer: 821M parameters)
- **📁 No compression needed** - direct PyTorch format
- **🗄️ HDF5 backups** for models >1GB with efficient compression
- **📊 Learning events** in PostgreSQL for analysis
- **🔍 File integrity** with automatic checksum verification

The system recognizes that **neural networks themselves contain the learned knowledge** through their weights and biases, while using PostgreSQL for structured learning event analysis.

---

*This modern hybrid architecture provides the most efficient, reliable foundation for TRUE AGI environmental learning.*
