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
        print(f"🔍 [NEO4J] DEBUG: connect() called")
        print(f"🔍 [NEO4J] DEBUG: URI: {self.uri}")
        print(f"🔍 [NEO4J] DEBUG: Username: {self.username}")
        print(f"🔍 [NEO4J] DEBUG: Database: {self.database}")
        
        try:
            # Suppress Neo4j warnings
            import logging
            logging.getLogger("neo4j").setLevel(logging.ERROR)
            
            print(f"🔍 [NEO4J] DEBUG: Creating driver...")
            self.driver = Neo4jGraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            print(f"🔍 [NEO4J] DEBUG: Driver created: {self.driver}")
            
            # Test connection
            print(f"🔍 [NEO4J] DEBUG: Testing connection...")
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 'Connection successful' AS message")
                test_result = result.single()
                print(f"🔍 [NEO4J] DEBUG: Connection test result: {test_result}")
            
            self.connected = True
            print(f"🔍 [NEO4J] DEBUG: Connection successful")
            return True
            
        except ServiceUnavailable as e:
            print(f"🔍 [NEO4J] DEBUG: ServiceUnavailable: {e}")
            print(f"🔍 [NEO4J] DEBUG: Cannot connect to Neo4j at {self.uri}")
            return False
        except AuthError as e:
            print(f"🔍 [NEO4J] DEBUG: AuthError: {e}")
            print(f"🔍 [NEO4J] DEBUG: Authentication failed for Neo4j")
            return False
        except Exception as e:
            print(f"🔍 [NEO4J] DEBUG: General connection error: {e}")
            print(f"🔍 [NEO4J] DEBUG: Error type: {type(e)}")
            return False
    
    def disconnect(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.connected = False
            # print("🔌 Disconnected from Neo4j")  # DISABLED TO REDUCE SPAM
    
    def _execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        print(f"🔍 [NEO4J] DEBUG: _execute_query called")
        print(f"🔍 [NEO4J] DEBUG: connected status: {self.connected}")
        print(f"🔍 [NEO4J] DEBUG: driver: {self.driver}")
        print(f"🔍 [NEO4J] DEBUG: query: {query[:100]}...")
        
        if not self.connected:
            print(f"🔍 [NEO4J] DEBUG: Not connected, raising ConnectionError")
            raise ConnectionError("Not connected to Neo4j database")
        
        if not self.driver:
            print(f"🔍 [NEO4J] DEBUG: No driver available, raising ConnectionError")
            raise ConnectionError("No Neo4j driver available")
        
        start_time = time.time()
        
        try:
            print(f"🔍 [NEO4J] DEBUG: Creating session with database: {self.database}")
            with self.driver.session(database=self.database) as session:
                print(f"🔍 [NEO4J] DEBUG: Running query with parameters: {parameters}")
                result = session.run(query, parameters or {})
                print(f"🔍 [NEO4J] DEBUG: Query executed, processing records...")
                records = [record.data() for record in result]
                print(f"🔍 [NEO4J] DEBUG: Processed {len(records)} records")
                
                self.stats['queries_executed'] += 1
                self.stats['last_operation_time'] = time.time() - start_time
                
                return records
                
        except Exception as e:
            print(f"🔍 [NEO4J] DEBUG: Query execution error: {e}")
            print(f"🔍 [NEO4J] DEBUG: Error type: {type(e)}")
            print(f"🔍 [NEO4J] DEBUG: Query: {query}")
            print(f"🔍 [NEO4J] DEBUG: Parameters: {parameters}")
            
            # Try to reconnect if connection was lost
            if "Connection" in str(e) or "connection" in str(e) or "Connection" in str(type(e)):
                print(f"🔍 [NEO4J] DEBUG: Detected connection error, trying to reconnect...")
                self.connected = False
                if self.connect():
                    print(f"🔍 [NEO4J] DEBUG: Reconnected successfully, retrying query...")
                    try:
                        with self.driver.session(database=self.database) as session:
                            result = session.run(query, parameters or {})
                            records = [record.data() for record in result]
                            print(f"🔍 [NEO4J] DEBUG: Retry successful, got {len(records)} records")
                            return records
                    except Exception as retry_e:
                        print(f"🔍 [NEO4J] DEBUG: Retry failed: {retry_e}")
                        return []
                else:
                    print(f"🔍 [NEO4J] DEBUG: Reconnection failed")
            
            return []
    
    def add_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Add a node to the Neo4j graph"""
        print(f"🔍 [NEO4J] DEBUG: add_node called with node_id='{node_id}'")
        print(f"🔍 [NEO4J] DEBUG: properties keys: {list(properties.keys()) if properties else 'None'}")
        print(f"🔍 [NEO4J] DEBUG: connected status: {self.connected}")
        
        if not self.connected:
            print(f"🔍 [NEO4J] DEBUG: Not connected, attempting to connect...")
            if not self.connect():
                print(f"🔍 [NEO4J] DEBUG: Connection failed")
                return False
            print(f"🔍 [NEO4J] DEBUG: Connection successful")
        
        # Sanitize properties for Neo4j
        print(f"🔍 [NEO4J] DEBUG: Sanitizing properties...")
        sanitized_props = self._sanitize_properties(properties)
        print(f"🔍 [NEO4J] DEBUG: Sanitized properties keys: {list(sanitized_props.keys()) if sanitized_props else 'None'}")
        
        query = """
        MERGE (n:Entity {id: $node_id})
        SET n += $properties
        RETURN n.id as id
        """
        
        print(f"🔍 [NEO4J] DEBUG: Executing query: {query}")
        print(f"🔍 [NEO4J] DEBUG: Query parameters: node_id='{node_id}', properties keys: {list(sanitized_props.keys())}")
        
        try:
            result = self._execute_query(query, {
                'node_id': node_id,
                'properties': sanitized_props
            })
            print(f"🔍 [NEO4J] DEBUG: Query result: {result}")
            
            if result:
                self.stats['nodes_created'] += 1
                print(f"🔍 [NEO4J] DEBUG: Node created successfully")
                return True
            else:
                print(f"🔍 [NEO4J] DEBUG: Query returned empty result")
                return False
        except Exception as e:
            print(f"🔍 [NEO4J] DEBUG: Query execution failed: {e}")
            print(f"🔍 [NEO4J] DEBUG: Exception type: {type(e)}")
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
    
    def find_patterns(self):
        """Find common patterns in the knowledge graph"""
        try:
            # Find triangles (3-node cycles)
            triangles = self._execute_query("""
                MATCH (a:Entity)-[r1]->(b:Entity)-[r2]->(c:Entity)-[r3]->(a)
                RETURN count(*) as triangles
            """)
            
            # Find chains (simple paths of length 3)
            chains = self._execute_query("""
                MATCH (a:Entity)-[r1]->(b:Entity)-[r2]->(c:Entity)-[r3]->(d:Entity)
                WHERE NOT (a)-[]->(d)
                RETURN count(*) as chains
            """)
            
            # Find cliques (nodes with many connections)
            cliques = self._execute_query("""
                MATCH (n:Entity)
                WHERE size((n)--()) >= 5
                RETURN count(n) as cliques
            """)
            
            # Find hub nodes (high degree centrality)
            hubs = self._execute_query("""
                MATCH (n:Entity)
                WHERE size((n)--()) >= 10
                RETURN count(n) as hubs
            """)
            
            patterns = []
            if triangles:
                patterns.extend([{'type': 'triangles', 'count': triangles[0]['triangles']}])
            if chains:
                patterns.extend([{'type': 'chains', 'count': chains[0]['chains']}])
            if cliques:
                patterns.extend([{'type': 'cliques', 'count': cliques[0]['cliques']}])
            if hubs:
                patterns.extend([{'type': 'hubs', 'count': hubs[0]['hubs']}])
            
            return patterns
            
        except Exception as e:
            return []
    
    def get_high_degree_nodes(self, threshold=10):
        """Get nodes with high connectivity"""
        try:
            result = self._execute_query("""
                MATCH (n:Entity)
                WHERE size((n)--()) >= $threshold
                RETURN n.id as id, size((n)--()) as degree
                ORDER BY degree DESC
                LIMIT 50
            """, {'threshold': threshold})
            
            return [record['id'] for record in result]
            
        except Exception as e:
            return []
    
    def get_node_properties(self, node_id):
        """Get properties of a specific node"""
        try:
            result = self._execute_query("""
                MATCH (n:Entity {id: $node_id})
                RETURN properties(n) as props
            """, {'node_id': node_id})
            
            return result[0]['props'] if result else {}
            
        except Exception as e:
            return {}
    
    def find_unexpected_connections(self):
        """Find unexpected or novel connections"""
        try:
            # Find nodes that are connected but have different labels
            result = self._execute_query("""
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE a.label <> b.label
                AND a.label IS NOT NULL
                AND b.label IS NOT NULL
                RETURN a.id as source, b.id as target, a.label as source_label, b.label as target_label
                LIMIT 20
            """)
            
            insights = []
            for record in result:
                insights.append({
                    'type': 'cross_label_connection',
                    'description': f"Cross-label connection: {record['source']} ({record['source_label']}) -> {record['target']} ({record['target_label']})"
                })
            
            return insights
            
        except Exception as e:
            return []
    
    def get_graph_stats(self):
        """Get comprehensive graph statistics"""
        try:
            # Get node and edge counts
            node_count = self._execute_query("MATCH (n:Entity) RETURN count(n) as count")[0]['count']
            edge_count = self._execute_query("MATCH (n:Entity)-[r]->() RETURN count(r) as count")[0]['count']
            
            # Get average degree
            avg_degree_result = self._execute_query("""
                MATCH (n:Entity)
                RETURN avg(size((n)--())) as avg_degree
            """)
            avg_degree = avg_degree_result[0]['avg_degree'] if avg_degree_result else 0
            
            # Get clustering coefficient (approximate)
            clustering_result = self._execute_query("""
                MATCH (n:Entity)
                OPTIONAL MATCH (n)-[r1]-(m:Entity), (n)-[r2]-(o:Entity)
                WHERE m <> o
                OPTIONAL MATCH (m)-[r3]-(o)
                WITH n, count(DISTINCT m) as neighbors, count(r3) as triangles
                WHERE neighbors > 1
                RETURN avg(toFloat(triangles) / (neighbors * (neighbors - 1))) as clustering
            """)
            clustering = clustering_result[0]['clustering'] if clustering_result else 0
            
            return {
                'total_nodes': node_count,
                'total_edges': edge_count,
                'average_degree': avg_degree,
                'clustering_coefficient': clustering or 0,
                'modularity': 0.5,  # Simplified
                'node_count': node_count,
                'edge_count': edge_count
            }
            
        except Exception as e:
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'average_degree': 0,
                'clustering_coefficient': 0,
                'modularity': 0,
                'node_count': 0,
                'edge_count': 0
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
                # print(f"✅ Index created: {query}")  # DISABLED TO REDUCE SPAM
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
