# 2-Database Architecture for ICA Framework

## Simple, Practical Solution

Instead of 4 different databases, we use just **2 databases** that each excel at their specific purpose:

### 1. Neo4j (Knowledge Graph)
- **Purpose**: Relationships and knowledge
- **Contains**: 
  - Concepts, hypotheses, causal relationships
  - Learning progress metadata
  - Session state
  - Neural model references (just metadata, not the models)
- **Why**: Excellent for graph relationships and knowledge representation

### 2. PostgreSQL (Neural Network Storage)
- **Purpose**: Large binary data storage
- **Contains**:
  - Neural network model weights (4GB+ files)
  - Training sessions and logs
  - Training metrics and history
- **Why**: Designed for large binary data, ACID compliance, proven scalability

## Benefits of This Approach

1. **Separation of Concerns**: Each database does what it's best at
2. **No Size Limits**: PostgreSQL handles multi-GB neural networks easily
3. **Simple Deployment**: Only 2 databases to manage
4. **Clear Architecture**: Graph data vs. binary data
5. **Proven Technology**: Both are enterprise-grade databases

## What Changed

- **Before**: Trying to store 4GB neural networks in Neo4j properties
- **After**: Neural networks in PostgreSQL, knowledge graph in Neo4j
- **Result**: No more "String header size out of range" errors

## Implementation Files

1. `postgresql_adapter.py` - PostgreSQL connection and operations
2. `simple_neural_persistence.py` - Simplified neural network storage
3. `two_database_config.json` - Configuration for both databases

## Next Steps

1. Install PostgreSQL
2. Update the main system to use the new persistence layer
3. Test the simplified approach
4. Deploy with confidence knowing each database handles its optimal data type

This solution is **much simpler** than the previous 4-database approach while being **more robust** than trying to store everything in Neo4j.
