"""
Enhanced Knowledge Graph for ICA Framework
Supports multiple database backends for scalable real-world applications
"""

from typing import Dict, List, Any, Optional, Union
import json
import time
from pathlib import Path
from dataclasses import is_dataclass

# Database adapters
from .database.graph_database import GraphDatabase
from .database.memory_adapter import MemoryAdapter

# Import Node and Edge classes for compatibility
try:
    from .components.causal_knowledge_graph import Node, Edge
    CAUSAL_CLASSES_AVAILABLE = True
except ImportError:
    CAUSAL_CLASSES_AVAILABLE = False

try:
    from .database.neo4j_adapter import Neo4jAdapter
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class EnhancedKnowledgeGraph:
    """
    Enhanced Knowledge Graph with pluggable database backends
    Supports both in-memory (NetworkX) and persistent (Neo4j) storage
    """
    
    def __init__(self, 
                 backend: str = 'memory',
                 config: Dict[str, Any] = None,
                 auto_connect: bool = True):
        """
        Initialize Enhanced Knowledge Graph
        
        Args:
            backend: Database backend ('memory' or 'neo4j')
            config: Database configuration
            auto_connect: Whether to auto-connect on initialization
        """
        
        self.backend_type = backend
        self.config = config or {}
        
        # Initialize database adapter
        self.db = self._create_database_adapter(backend, self.config)
        
        # Statistics tracking
        self.operation_stats = {
            'nodes_added': 0,
            'edges_added': 0,
            'queries_executed': 0,
            'total_operations': 0,
            'session_start_time': time.time()
        }
        
        # Auto-connect if requested
        if auto_connect:
            self.connect()
    
    def _create_database_adapter(self, backend: str, config: Dict[str, Any]) -> GraphDatabase:
        """Create appropriate database adapter"""
        
        if backend == 'memory':
            return MemoryAdapter(config)
        
        elif backend == 'neo4j':
            if not NEO4J_AVAILABLE:
                print("⚠️ Neo4j not available, falling back to memory adapter")
                return MemoryAdapter(config)
            return Neo4jAdapter(config)
        
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def connect(self) -> bool:
        """Establish database connection"""
        success = self.db.connect()
        
        if success and self.backend_type == 'neo4j':
            # Create indexes for better performance
            try:
                self.db.create_indexes()
            except AttributeError:
                pass  # Method not available in all adapters
        
        return success
    
    def disconnect(self):
        """Close database connection"""
        self.db.disconnect()
    
    def add_entity(self, entity_id: str, label: str = None, properties: Dict[str, Any] = None) -> bool:
        """Add an entity (node) to the knowledge graph"""
        entity_props = properties or {}
        
        # Add label if provided
        if label:
            entity_props['label'] = label
        
        # Ensure ID is in properties
        entity_props['id'] = entity_id
        
        success = self.db.add_node(entity_id, entity_props)
        
        if success:
            self.operation_stats['nodes_added'] += 1
            self.operation_stats['total_operations'] += 1
        
        return success
    
    def add_relationship(self, 
                        source: str, 
                        target: str, 
                        relationship_type: str,
                        confidence: float = 1.0,
                        properties: Dict[str, Any] = None) -> bool:
        """Add a relationship (edge) between entities"""
        
        rel_props = properties or {}
        rel_props['confidence'] = confidence
        rel_props['created_at'] = time.time()
        
        success = self.db.add_edge(source, target, relationship_type, rel_props)
        
        if success:
            self.operation_stats['edges_added'] += 1
            self.operation_stats['total_operations'] += 1
        
        return success
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an entity by ID"""
        self.operation_stats['queries_executed'] += 1
        return self.db.get_node(entity_id)
    
    def get_relationships(self, 
                         source: str = None,
                         target: str = None,
                         relationship_type: str = None) -> List[Dict[str, Any]]:
        """Get relationships with optional filtering"""
        self.operation_stats['queries_executed'] += 1
        return self.db.get_edges(source, target, relationship_type)
    
    def get_related_entities(self, entity_id: str, direction: str = 'both') -> List[str]:
        """Get entities related to a given entity"""
        self.operation_stats['queries_executed'] += 1
        return self.db.get_neighbors(entity_id, direction)
    
    def get_neighbors(self, entity_id: str, direction: str = 'both') -> List[str]:
        """Alias for get_related_entities for compatibility"""
        return self.get_related_entities(entity_id, direction)
    
    def entity_exists(self, entity_id: str) -> bool:
        """Check if an entity exists"""
        return self.db.node_exists(entity_id)
    
    def relationship_exists(self, source: str, target: str, relationship_type: str = None) -> bool:
        """Check if a relationship exists"""
        return self.db.edge_exists(source, target, relationship_type)
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update entity properties"""
        properties['updated_at'] = time.time()
        return self.db.update_node_properties(entity_id, properties)
    
    def update_relationship(self, 
                           source: str, 
                           target: str, 
                           relationship_type: str,
                           properties: Dict[str, Any]) -> bool:
        """Update relationship properties"""
        properties['updated_at'] = time.time()
        return self.db.update_edge_properties(source, target, relationship_type, properties)
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all its relationships"""
        return self.db.delete_node(entity_id)
    
    def delete_relationship(self, source: str, target: str, relationship_type: str = None) -> bool:
        """Delete a relationship"""
        return self.db.delete_edge(source, target, relationship_type)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge graph statistics"""
        db_stats = self.db.get_graph_stats()
        
        # Combine with operation stats
        session_time = time.time() - self.operation_stats['session_start_time']
        
        return {
            'database': db_stats,
            'session': {
                **self.operation_stats,
                'session_duration': session_time,
                'operations_per_second': self.operation_stats['total_operations'] / max(session_time, 0.001)
            },
            'backend': self.backend_type,
            'connection_status': self.db.connected
        }
    
    def clear(self) -> bool:
        """Clear all entities and relationships"""
        success = self.db.clear_graph()
        
        if success:
            # Reset operation stats
            self.operation_stats = {
                'nodes_added': 0,
                'edges_added': 0,
                'queries_executed': 0,
                'total_operations': 0,
                'session_start_time': time.time()
            }
        
        return success
    
    def backup(self, backup_path: Union[str, Path]) -> bool:
        """Create a backup of the knowledge graph"""
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        return self.db.backup_graph(str(backup_path))
    
    def restore(self, backup_path: Union[str, Path]) -> bool:
        """Restore knowledge graph from backup"""
        return self.db.restore_graph(str(backup_path))
    
    def export_to_file(self, file_path: Union[str, Path], format: str = 'json') -> bool:
        """Export knowledge graph to file"""
        try:
            graph_data = self.db.export_graph(format)
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(graph_data)
            
            return True
            
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def import_from_file(self, file_path: Union[str, Path], format: str = 'json') -> bool:
        """Import knowledge graph from file"""
        try:
            file_path = Path(file_path)
            
            with open(file_path, 'r') as f:
                graph_data = f.read()
            
            return self.db.import_graph(graph_data, format)
            
        except Exception as e:
            print(f"Import failed: {e}")
            return False
    
    def execute_custom_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom database query"""
        self.operation_stats['queries_executed'] += 1
        return self.db.execute_query(query, parameters)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        if hasattr(self.db, 'get_performance_metrics'):
            return self.db.get_performance_metrics()
        else:
            return self.get_statistics()
    
    def migrate_to_neo4j(self, neo4j_config: Dict[str, Any]) -> bool:
        """Migrate current graph to Neo4j database"""
        if not NEO4J_AVAILABLE:
            print("❌ Neo4j driver not available")
            return False
        
        # Export current graph
        current_data = self.db.export_graph('json')
        
        # Create new Neo4j adapter
        neo4j_db = Neo4jAdapter(neo4j_config)
        
        if not neo4j_db.connect():
            print("❌ Failed to connect to Neo4j")
            return False
        
        # Import data into Neo4j
        success = neo4j_db.import_graph(current_data, 'json')
        
        if success:
            # Switch to Neo4j backend
            self.db.disconnect()
            self.db = neo4j_db
            self.backend_type = 'neo4j'
            print("✅ Successfully migrated to Neo4j")
        else:
            neo4j_db.disconnect()
            print("❌ Migration to Neo4j failed")
        
        return success
    
    def switch_backend(self, backend: str, config: Dict[str, Any] = None) -> bool:
        """Switch to a different database backend"""
        if backend == self.backend_type:
            print(f"Already using {backend} backend")
            return True
        
        # Export current data
        try:
            current_data = self.db.export_graph('json')
        except Exception as e:
            print(f"Failed to export current data: {e}")
            return False
        
        # Create new database adapter
        new_config = config or self.config
        new_db = self._create_database_adapter(backend, new_config)
        
        if not new_db.connect():
            print(f"Failed to connect to {backend} backend")
            return False
        
        # Import data into new backend
        success = new_db.import_graph(current_data, 'json')
        
        if success:
            # Switch backends
            self.db.disconnect()
            self.db = new_db
            self.backend_type = backend
            self.config = new_config
            print(f"✅ Successfully switched to {backend} backend")
        else:
            new_db.disconnect()
            print(f"❌ Failed to switch to {backend} backend")
        
        return success
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get schema information about the knowledge graph"""
        stats = self.get_statistics()
        
        # Get sample entities and relationships
        sample_entities = []
        sample_relationships = []
        
        try:
            # Get a few sample entities
            if self.backend_type == 'memory' and hasattr(self.db, 'graph'):
                nodes = list(self.db.graph.nodes())[:5]
                for node in nodes:
                    entity = self.get_entity(node)
                    if entity:
                        sample_entities.append(entity)
                
                edges = list(self.db.graph.edges(keys=True))[:5]
                for source, target, rel_type in edges:
                    relationships = self.get_relationships(source, target, rel_type)
                    if relationships:
                        sample_relationships.extend(relationships[:1])
        
        except Exception as e:
            print(f"Error getting schema samples: {e}")
        
        return {
            'statistics': stats,
            'sample_entities': sample_entities,
            'sample_relationships': sample_relationships,
            'backend': self.backend_type,
            'relationship_types': stats['database'].get('relationship_type_list', [])
        }
    
    def import_from_networkx(self, nx_graph) -> bool:
        """Import data from a NetworkX graph into the enhanced knowledge graph"""
        try:
            import networkx as nx
            
            if not isinstance(nx_graph, nx.Graph):
                print(f"⚠️ Expected NetworkX graph, got {type(nx_graph)}")
                return False
            
            print(f"📊 Importing {nx_graph.number_of_nodes()} nodes and {nx_graph.number_of_edges()} edges from NetworkX...")
            
            # Import nodes
            nodes_imported = 0
            for node_id, node_data in nx_graph.nodes(data=True):
                # Extract label from properties if available
                label = node_data.get('label', 'entity')
                
                # Create properties dict
                properties = dict(node_data)
                if 'label' in properties:
                    del properties['label']  # Don't duplicate label in properties
                
                if self.add_entity(str(node_id), label, properties):
                    nodes_imported += 1
            
            # Import edges
            edges_imported = 0
            for source, target, edge_data in nx_graph.edges(data=True):
                # Extract relationship type and confidence
                rel_type = edge_data.get('type', 'related')
                confidence = edge_data.get('confidence', 1.0)
                
                # Create properties dict
                properties = dict(edge_data)
                if 'type' in properties:
                    del properties['type']
                if 'confidence' in properties:
                    del properties['confidence']
                
                if self.add_relationship(str(source), str(target), rel_type, confidence, properties):
                    edges_imported += 1
            
            print(f"✅ Import complete: {nodes_imported} nodes, {edges_imported} edges")
            return True
            
        except Exception as e:
            print(f"❌ Error importing from NetworkX: {e}")
            return False

    # Compatibility methods for existing ICA Framework code
    def add_node(self, node_id_or_node, **properties) -> bool:
        """Compatibility method for ICA Framework - maps to add_entity"""
        try:
            # Check if it's a dataclass Node object
            if is_dataclass(node_id_or_node) and hasattr(node_id_or_node, 'id'):
                node = node_id_or_node
                node_id = node.id
                label = node.label
                # Combine all properties
                all_properties = {}
                all_properties.update(node.properties_static or {})
                all_properties.update(node.properties_dynamic or {})
                all_properties['confidence'] = node.confidence
                return self.add_entity(node_id, label, all_properties)
            else:
                # Direct call with node_id
                label = properties.pop('label', None)
                return self.add_entity(str(node_id_or_node), label, properties)
        except Exception as e:
            print(f"❌ Error in add_node: {e}")
            return False
    
    def add_edge(self, source_or_edge, target=None, relationship_type='related', **properties) -> bool:
        """Compatibility method for ICA Framework - maps to add_relationship"""
        try:
            # Check if it's a dataclass Edge object
            if is_dataclass(source_or_edge) and hasattr(source_or_edge, 'source'):
                edge = source_or_edge
                source = edge.source
                target = edge.target
                relationship_type = edge.relationship
                confidence = edge.confidence
                edge_properties = edge.properties.copy() if edge.properties else {}
                edge_properties['weight'] = edge.weight
                edge_properties.update(edge.conditions or {})
                return self.add_relationship(source, target, relationship_type, confidence, edge_properties)
            else:
                # Direct call
                confidence = properties.pop('confidence', 1.0)
                return self.add_relationship(str(source_or_edge), str(target), relationship_type, confidence, properties)
        except Exception as e:
            print(f"❌ Error in add_edge: {e}")
            return False
    
    def has_node(self, node_id: str) -> bool:
        """Compatibility method for NetworkX-style node existence check"""
        return self.entity_exists(node_id)
    
    def has_edge(self, source: str, target: str) -> bool:
        """Compatibility method for NetworkX-style edge existence check"""
        return self.relationship_exists(source, target)
    
    def neighbors(self, node_id: str) -> List[str]:
        """Compatibility method for NetworkX-style neighbors access"""
        return self.get_neighbors(node_id)
    
    def number_of_nodes(self) -> int:
        """Compatibility method for NetworkX-style node count"""
        stats = self.get_statistics()
        return stats['database'].get('node_count', 0)
    
    def number_of_edges(self) -> int:
        """Compatibility method for NetworkX-style edge count"""
        stats = self.get_statistics()
        return stats['database'].get('edge_count', 0)

    @property 
    def nodes_dict(self):
        """Compatibility property for nodes_dict access"""
        class NodesDict:
            def __init__(self, kg):
                self.kg = kg
            
            def keys(self):
                try:
                    result = self.kg.execute_custom_query("MATCH (n) RETURN n.id as id")
                    return [record['id'] for record in result]
                except:
                    return []
        
        return NodesDict(self)

    @property 
    def graph(self):
        """Compatibility property for NetworkX graph access"""
        if hasattr(self.db, 'get_networkx_graph'):
            return self.db.get_networkx_graph()
        else:
            # For non-NetworkX backends, create a minimal interface
            class GraphInterface:
                def __init__(self, kg):
                    self.kg = kg
                
                def number_of_nodes(self):
                    try:
                        return self.kg.db.get_node_count()
                    except:
                        stats = self.kg.get_statistics()
                        return stats['database'].get('node_count', 0)
                
                def number_of_edges(self):
                    try:
                        return self.kg.db.get_edge_count()
                    except:
                        stats = self.kg.get_statistics()
                        return stats['database'].get('edge_count', 0)
                
                def nodes(self, data=False):
                    # This is a simplified implementation
                    # Real implementation would need to query all nodes
                    return []
                
                def edges(self, data=False):
                    # This is a simplified implementation
                    return []
            
            return GraphInterface(self)
    
    def get_stats(self) -> Dict[str, Any]:
        """Backward compatibility method for get_statistics"""
        stats = self.get_statistics()
        # Return a simplified format that matches expected interface
        # Handle both 'nodes'/'edges' and 'node_count'/'edge_count' formats
        db_stats = stats.get('database', {})
        return {
            'nodes': db_stats.get('nodes', db_stats.get('node_count', 0)),
            'edges': db_stats.get('edges', db_stats.get('edge_count', 0)),
            'backend': self.backend_type,
            'connected': self.db.connected if hasattr(self.db, 'connected') else True
        }
    
    def update_edge_confidence(self, edge_id: str, success: bool) -> bool:
        """Update edge confidence based on success/failure"""
        # For now, this is a simplified implementation
        # In a real system, you'd need to track edge IDs more carefully
        try:
            # Get all edges and update confidence
            # This is a simplified approach - in practice you'd want better edge tracking
            return True
        except Exception as e:
            return False
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        return self.get_statistics()
    
    def save(self, file_path: str) -> bool:
        """Save knowledge graph to file (compatibility method)"""
        return self.export_to_file(file_path, 'json')
    
    def load(self, file_path: str) -> bool:
        """Load knowledge graph from file (compatibility method)"""
        return self.import_from_file(file_path, 'json')
