"""
Neo4j adapter for ICA Framework knowledge graph storage
Provides high-performance graph database backend for real-world scenarios
"""

from typing import Dict, List, Any, Optional
import json
import time
from .graph_database import GraphDatabase

try:
    from neo4j import GraphDatabase as Neo4jGraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Neo4j driver not available. Install with: pip install neo4j")


class Neo4jAdapter(GraphDatabase):
    """Neo4j implementation of the graph database interface"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not installed. Run: pip install neo4j")
        
        # Default configuration
        self.uri = config.get('uri', 'bolt://localhost:7687')
        self.username = config.get('username', 'neo4j')
        self.password = config.get('password', 'password')
        self.database = config.get('database', 'neo4j')
        
        self.driver = None
        self.session = None
        
        # Performance tracking
        self.stats = {
            'queries_executed': 0,
            'nodes_created': 0,
            'edges_created': 0,
            'last_operation_time': 0
        }
    
    def connect(self) -> bool:
        """Establish connection to Neo4j database"""
        try:
            self.driver = Neo4jGraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 'Connection successful' AS message")
                result.single()
            
            self.connected = True
            print(f"✅ Connected to Neo4j at {self.uri}")
            return True
            
        except ServiceUnavailable:
            print(f"❌ Cannot connect to Neo4j at {self.uri}")
            return False
        except AuthError:
            print(f"❌ Authentication failed for Neo4j")
            return False
        except Exception as e:
            print(f"❌ Neo4j connection error: {e}")
            return False
    
    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.connected = False
            print("🔌 Disconnected from Neo4j")
    
    def _execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        if not self.connected:
            raise ConnectionError("Not connected to Neo4j database")
        
        start_time = time.time()
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]
                
                self.stats['queries_executed'] += 1
                self.stats['last_operation_time'] = time.time() - start_time
                
                return records
                
        except Exception as e:
            print(f"Query execution error: {e}")
            print(f"Query: {query}")
            print(f"Parameters: {parameters}")
            return []
    
    def add_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Add a node to the Neo4j graph"""
        # Sanitize properties for Neo4j
        sanitized_props = self._sanitize_properties(properties)
        
        query = """
        MERGE (n:Entity {id: $node_id})
        SET n += $properties
        RETURN n.id as id
        """
        
        result = self._execute_query(query, {
            'node_id': node_id,
            'properties': sanitized_props
        })
        
        if result:
            self.stats['nodes_created'] += 1
            return True
        return False
    
    def add_edge(self, source: str, target: str, relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        """Add an edge between two nodes in Neo4j"""
        sanitized_props = self._sanitize_properties(properties or {})
        
        # Neo4j relationship type must be valid identifier
        rel_type = self._sanitize_relationship_type(relationship_type)
        
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r += $properties
        RETURN r
        """
        
        result = self._execute_query(query, {
            'source_id': source,
            'target_id': target,
            'properties': sanitized_props
        })
        
        if result:
            self.stats['edges_created'] += 1
            return True
        return False
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by ID from Neo4j"""
        query = """
        MATCH (n:Entity {id: $node_id})
        RETURN properties(n) as properties
        """
        
        result = self._execute_query(query, {'node_id': node_id})
        return result[0]['properties'] if result else None
    
    def get_edges(self, source: str = None, target: str = None, relationship_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve edges with optional filtering from Neo4j"""
        conditions = []
        params = {}
        
        if source:
            conditions.append("source.id = $source_id")
            params['source_id'] = source
        
        if target:
            conditions.append("target.id = $target_id")
            params['target_id'] = target
        
        rel_pattern = "r"
        if relationship_type:
            rel_type = self._sanitize_relationship_type(relationship_type)
            rel_pattern = f"r:{rel_type}"
        
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
        MATCH (source:Entity)-[{rel_pattern}]->(target:Entity)
        {where_clause}
        RETURN source.id as source, target.id as target, type(r) as relationship_type, properties(r) as properties
        """
        
        return self._execute_query(query, params)
    
    def get_neighbors(self, node_id: str, direction: str = 'both') -> List[str]:
        """Get neighboring nodes from Neo4j"""
        if direction == 'outgoing':
            pattern = "(n:Entity {id: $node_id})-[]->(neighbor:Entity)"
        elif direction == 'incoming':
            pattern = "(neighbor:Entity)-[]->(n:Entity {id: $node_id})"
        else:  # both
            pattern = "(n:Entity {id: $node_id})-[]-(neighbor:Entity)"
        
        query = f"""
        MATCH {pattern}
        RETURN DISTINCT neighbor.id as neighbor_id
        """
        
        result = self._execute_query(query, {'node_id': node_id})
        return [record['neighbor_id'] for record in result]
    
    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists in Neo4j"""
        query = """
        MATCH (n:Entity {id: $node_id})
        RETURN count(n) > 0 as exists
        """
        
        result = self._execute_query(query, {'node_id': node_id})
        return result[0]['exists'] if result else False
    
    def edge_exists(self, source: str, target: str, relationship_type: str = None) -> bool:
        """Check if an edge exists in Neo4j"""
        rel_pattern = "r"
        if relationship_type:
            rel_type = self._sanitize_relationship_type(relationship_type)
            rel_pattern = f"r:{rel_type}"
        
        query = f"""
        MATCH (source:Entity {{id: $source_id}})-[{rel_pattern}]->(target:Entity {{id: $target_id}})
        RETURN count(r) > 0 as exists
        """
        
        result = self._execute_query(query, {
            'source_id': source,
            'target_id': target
        })
        return result[0]['exists'] if result else False
    
    def get_node_count(self) -> int:
        """Get total number of nodes in Neo4j"""
        query = "MATCH (n:Entity) RETURN count(n) as count"
        result = self._execute_query(query)
        return result[0]['count'] if result else 0
    
    def get_edge_count(self) -> int:
        """Get total number of edges in Neo4j"""
        query = "MATCH ()-[r]->() RETURN count(r) as count"
        result = self._execute_query(query)
        return result[0]['count'] if result else 0
    
    def update_node_properties(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties in Neo4j"""
        sanitized_props = self._sanitize_properties(properties)
        
        query = """
        MATCH (n:Entity {id: $node_id})
        SET n += $properties
        RETURN n.id as id
        """
        
        result = self._execute_query(query, {
            'node_id': node_id,
            'properties': sanitized_props
        })
        return bool(result)
    
    def update_edge_properties(self, source: str, target: str, relationship_type: str, properties: Dict[str, Any]) -> bool:
        """Update edge properties in Neo4j"""
        sanitized_props = self._sanitize_properties(properties)
        rel_type = self._sanitize_relationship_type(relationship_type)
        
        query = f"""
        MATCH (source:Entity {{id: $source_id}})-[r:{rel_type}]->(target:Entity {{id: $target_id}})
        SET r += $properties
        RETURN r
        """
        
        result = self._execute_query(query, {
            'source_id': source,
            'target_id': target,
            'properties': sanitized_props
        })
        return bool(result)
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges from Neo4j"""
        query = """
        MATCH (n:Entity {id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        
        result = self._execute_query(query, {'node_id': node_id})
        return result[0]['deleted_count'] > 0 if result else False
    
    def delete_edge(self, source: str, target: str, relationship_type: str = None) -> bool:
        """Delete an edge from Neo4j"""
        rel_pattern = "r"
        if relationship_type:
            rel_type = self._sanitize_relationship_type(relationship_type)
            rel_pattern = f"r:{rel_type}"
        
        query = f"""
        MATCH (source:Entity {{id: $source_id}})-[{rel_pattern}]->(target:Entity {{id: $target_id}})
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = self._execute_query(query, {
            'source_id': source,
            'target_id': target
        })
        return result[0]['deleted_count'] > 0 if result else False
    
    def clear_graph(self) -> bool:
        """Clear all nodes and edges from Neo4j"""
        query = """
        MATCH (n)
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """
        
        result = self._execute_query(query)
        return result[0]['deleted_count'] >= 0 if result else False
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query"""
        return self._execute_query(query, parameters)
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics from Neo4j"""
        stats_query = """
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]->()
        RETURN 
            count(DISTINCT n) as node_count,
            count(r) as edge_count,
            count(DISTINCT type(r)) as relationship_types,
            collect(DISTINCT type(r)) as relationship_type_list
        """
        
        result = self._execute_query(stats_query)
        
        if result:
            db_stats = result[0]
            return {
                'nodes': db_stats['node_count'],
                'edges': db_stats['edge_count'],
                'relationship_types': db_stats['relationship_types'],
                'relationship_type_list': db_stats['relationship_type_list'],
                'performance': self.stats,
                'database': self.database,
                'connected': self.connected
            }
        
        return {
            'nodes': 0,
            'edges': 0,
            'relationship_types': 0,
            'relationship_type_list': [],
            'performance': self.stats,
            'database': self.database,
            'connected': self.connected
        }
    
    def export_graph(self, format: str = 'json') -> str:
        """Export graph data from Neo4j"""
        if format != 'json':
            raise ValueError("Only JSON format supported for Neo4j export")
        
        # Export nodes
        nodes_query = """
        MATCH (n:Entity)
        RETURN n.id as id, properties(n) as properties
        """
        
        # Export edges
        edges_query = """
        MATCH (source:Entity)-[r]->(target:Entity)
        RETURN source.id as source, target.id as target, type(r) as relationship_type, properties(r) as properties
        """
        
        nodes_result = self._execute_query(nodes_query)
        edges_result = self._execute_query(edges_query)
        
        export_data = {
            'nodes': nodes_result,
            'edges': edges_result,
            'metadata': {
                'export_time': time.time(),
                'database': self.database,
                'stats': self.get_graph_stats()
            }
        }
        
        return json.dumps(export_data, indent=2)
    
    def import_graph(self, data: str, format: str = 'json') -> bool:
        """Import graph data into Neo4j"""
        if format != 'json':
            raise ValueError("Only JSON format supported for Neo4j import")
        
        try:
            import_data = json.loads(data)
            
            # Clear existing graph
            self.clear_graph()
            
            # Import nodes
            for node in import_data.get('nodes', []):
                self.add_node(node['id'], node['properties'])
            
            # Import edges
            for edge in import_data.get('edges', []):
                self.add_edge(
                    edge['source'],
                    edge['target'],
                    edge['relationship_type'],
                    edge['properties']
                )
            
            return True
            
        except Exception as e:
            print(f"Import failed: {e}")
            return False
    
    def _sanitize_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize properties for Neo4j storage"""
        sanitized = {}
        
        for key, value in properties.items():
            # Convert numpy types to native Python types
            if hasattr(value, 'item'):
                value = value.item()
            elif hasattr(value, 'tolist'):
                value = value.tolist()
            
            # Neo4j doesn't support nested objects, convert to string
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            # Ensure key is valid Neo4j property name
            clean_key = key.replace(' ', '_').replace('-', '_')
            sanitized[clean_key] = value
        
        return sanitized
    
    def _sanitize_relationship_type(self, rel_type: str) -> str:
        """Sanitize relationship type for Neo4j"""
        # Neo4j relationship types must be valid identifiers
        return rel_type.upper().replace(' ', '_').replace('-', '_')
    
    def create_indexes(self):
        """Create performance indexes on the Neo4j database"""
        index_queries = [
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (n:Entity) ON (n.id)",
            "CREATE INDEX entity_label_index IF NOT EXISTS FOR (n:Entity) ON (n.label)",
        ]
        
        for query in index_queries:
            try:
                self._execute_query(query)
                print(f"✅ Index created: {query}")
            except Exception as e:
                print(f"⚠️ Index creation failed: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'connection_status': self.connected,
            'database_stats': self.get_graph_stats(),
            'operation_stats': self.stats,
            'database_info': {
                'uri': self.uri,
                'database': self.database,
                'driver_available': NEO4J_AVAILABLE
            }
        }
