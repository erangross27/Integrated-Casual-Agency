# Database Backend for ICA Framework

The ICA Framework now supports multiple database backends for knowledge graph storage, enabling real-world scalable applications.

## Supported Backends

### 1. Memory Backend (Default)
- **Use case**: Development, testing, small datasets
- **Technology**: NetworkX in-memory graphs
- **Pros**: Fast, no setup required, built-in
- **Cons**: Limited by RAM, data lost on restart

### 2. Neo4j Backend
- **Use case**: Production, large datasets, persistent storage
- **Technology**: Neo4j graph database
- **Pros**: Persistent, scalable, powerful queries, ACID transactions
- **Cons**: Requires Neo4j installation

## Quick Start

### Using Memory Backend (Default)
```python
from examples.learning import ContinuousLearning

# Uses in-memory storage
learner = ContinuousLearning()
learner.run_continuous_learning()
```

### Using Neo4j Backend
```python
from examples.learning import ContinuousLearning

# Configure Neo4j
neo4j_config = {
    'uri': 'bolt://localhost:7687',
    'username': 'neo4j',
    'password': 'your_password',
    'database': 'neo4j'
}

# Use Neo4j backend
learner = ContinuousLearning(
    database_backend='neo4j',
    database_config=neo4j_config
)
learner.run_continuous_learning()
```

### Command Line Usage
```bash
# Use memory backend (default)
python examples/learning.py

# Use Neo4j backend
python examples/learning.py --backend neo4j --neo4j-password your_password

# Full Neo4j configuration
python examples/learning.py \
  --backend neo4j \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password your_password \
  --neo4j-database ica_knowledge
```

## Installation

### Core Requirements (Always Needed)
```bash
pip install numpy networkx
```

### Neo4j Support (Optional)
```bash
# Install Neo4j Python driver (included in core requirements)
pip install -r requirements/requirements.txt

# Or install just the Neo4j driver
pip install neo4j
```

### Interactive Setup
```bash
# Run the database setup wizard
python setup_database.py
```

## Neo4j Setup

### 1. Install Neo4j
Download and install Neo4j Desktop or Community Edition:
- **Neo4j Desktop**: https://neo4j.com/download/
- **Docker**: `docker run -p 7474:7474 -p 7687:7687 neo4j`

### 2. Configure Neo4j
1. Start Neo4j database
2. Open Neo4j Browser (http://localhost:7474)
3. Set username/password (default: neo4j/neo4j)
4. Create a new database (optional)

### 3. Test Connection
```python
python setup_database.py
# Choose option 4: Test Neo4j connection
```

## Real-World Benefits

### Scalability Comparison

| Feature | Memory Backend | Neo4j Backend |
|---------|----------------|---------------|
| **Nodes** | ~1M (RAM limited) | Billions |
| **Edges** | ~10M (RAM limited) | Billions |
| **Persistence** | ❌ Lost on restart | ✅ Persistent |
| **Concurrent Access** | ❌ Single process | ✅ Multi-user |
| **ACID Transactions** | ❌ | ✅ |
| **Query Language** | Python/NetworkX | Cypher |
| **Performance** | Very fast | Fast + optimized |

### Real-World Use Cases

#### Development & Testing
```python
# Quick prototyping with memory backend
learner = ContinuousLearning(database_backend='memory')
```

#### Production IoT System
```python
# Persistent storage for continuous learning
learner = ContinuousLearning(
    database_backend='neo4j',
    database_config={
        'uri': 'bolt://production-server:7687',
        'username': 'ica_agent',
        'password': 'secure_password',
        'database': 'iot_knowledge_graph'
    }
)
```

#### Multi-Agent Learning
```python
# Multiple agents sharing knowledge via Neo4j
config = {'uri': 'bolt://shared-db:7687', ...}

agent1 = ContinuousLearning(database_backend='neo4j', database_config=config)
agent2 = ContinuousLearning(database_backend='neo4j', database_config=config)
# Both agents contribute to the same knowledge graph
```

## Migration

### From Memory to Neo4j
```python
# Start with memory backend
learner = ContinuousLearning(database_backend='memory')
learner.run_continuous_learning()  # Learn some knowledge

# Migrate to Neo4j for persistence
neo4j_config = {...}
success = learner.agent.knowledge_graph.migrate_to_neo4j(neo4j_config)

if success:
    print("✅ Successfully migrated to Neo4j!")
```

### Backup and Restore
```python
# Backup current knowledge graph
learner.agent.knowledge_graph.backup('backup/knowledge_graph_20250712.json')

# Restore from backup
learner.agent.knowledge_graph.restore('backup/knowledge_graph_20250712.json')
```

## Performance Tips

### Memory Backend
- Monitor RAM usage with large graphs
- Use for development and testing
- Consider switching to Neo4j when nodes > 100K

### Neo4j Backend
- Create indexes for better performance (automatic)
- Use connection pooling for multiple agents
- Monitor Neo4j memory settings
- Use SSD storage for better I/O performance

### Query Optimization
```python
# Efficient entity lookup
entity = kg.get_entity('sensor_123')

# Efficient relationship queries
relationships = kg.get_relationships(source='sensor_123', relationship_type='measures')

# Custom Cypher queries for complex operations
results = kg.execute_custom_query("""
    MATCH (sensor:Entity {label: 'sensor'})-[:measures]->(env:Entity {label: 'environment'})
    RETURN sensor.id, env.id, count(*) as connections
""")
```

## Monitoring

### Statistics
```python
# Get comprehensive statistics
stats = learner.agent.knowledge_graph.get_statistics()
print(f"Nodes: {stats['database']['nodes']}")
print(f"Edges: {stats['database']['edges']}")
print(f"Backend: {stats['backend']}")
print(f"Operations/sec: {stats['session']['operations_per_second']}")
```

### Performance Metrics
```python
# Detailed performance metrics
metrics = learner.agent.knowledge_graph.get_performance_metrics()
print(f"Connection status: {metrics['connection_status']}")
print(f"Query performance: {metrics['operation_stats']}")
```

## Troubleshooting

### Common Issues

#### Neo4j Connection Failed
```
❌ Cannot connect to Neo4j at bolt://localhost:7687
```
**Solution**: 
1. Ensure Neo4j is running
2. Check connection details
3. Verify firewall settings

#### Memory Issues with Large Graphs
```
MemoryError: Unable to allocate array
```
**Solution**: Switch to Neo4j backend for large-scale learning

#### Import/Export Errors
```
Import failed: Invalid JSON format
```
**Solution**: Ensure backup files are valid JSON format

### Performance Monitoring
```bash
# Monitor Neo4j performance
# Open Neo4j Browser: http://localhost:7474
# Run: :server status
```

## Future Enhancements

Planned database backends:
- **MongoDB**: Document-based knowledge storage
- **Redis**: High-speed in-memory with persistence
- **ArangoDB**: Multi-model database
- **Amazon Neptune**: Cloud-native graph database

## Examples

See the `examples/` directory for complete examples:
- `learning.py`: Enhanced with database backend support
- `test_learning.py`: Quick testing script
- Database configuration examples in `config/database/`
