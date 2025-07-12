"""
In-memory adapter for ICA Framework knowledge graph storage
Provides fast in-memory graph database for development and testing
"""

from typing import Dict, List, Any, Optional
import json
import time
import networkx as nx
from .graph_database import GraphDatabase


class MemoryAdapter(GraphDatabase):
    """In-memory implementation using NetworkX for fast development"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.graph = nx.MultiDiGraph()  # Support multiple edges between nodes
        self.node_properties = {}  # Store additional node properties
        self.edge_properties = {}  # Store edge properties
        
        # Performance tracking
        self.stats = {
            'queries_executed': 0,
            'nodes_created': 0,
            'edges_created': 0,
            'last_operation_time': 0
        }
    
    def connect(self) -> bool:
        """Establish connection (always successful for memory adapter)"""
        self.connected = True
        print("✅ Memory adapter connected")
        return True
    
    def disconnect(self):
        """Close connection"""
        self.connected = False
        print("🔌 Memory adapter disconnected")
    
    def add_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Add a node to the in-memory graph"""
        start_time = time.time()
        
        try:
            self.graph.add_node(node_id)
            self.node_properties[node_id] = properties.copy()
            
            self.stats['nodes_created'] += 1
            self.stats['last_operation_time'] = time.time() - start_time
            return True
            
        except Exception as e:
            print(f"Error adding node: {e}")
            return False
    
    def add_edge(self, source: str, target: str, relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        """Add an edge between two nodes"""
        start_time = time.time()
        
        try:
            # Ensure nodes exist
            if not self.node_exists(source):
                self.add_node(source, {'id': source})
            if not self.node_exists(target):
                self.add_node(target, {'id': target})
            
            # Add edge with key as relationship type
            self.graph.add_edge(source, target, key=relationship_type)
            
            # Store edge properties
            edge_key = f"{source}->{target}:{relationship_type}"
            self.edge_properties[edge_key] = (properties or {}).copy()
            
            self.stats['edges_created'] += 1
            self.stats['last_operation_time'] = time.time() - start_time
            return True
            
        except Exception as e:
            print(f"Error adding edge: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by ID"""
        start_time = time.time()
        
        if node_id in self.node_properties:
            self.stats['queries_executed'] += 1
            self.stats['last_operation_time'] = time.time() - start_time
            return self.node_properties[node_id].copy()
        
        return None
    
    def get_edges(self, source: str = None, target: str = None, relationship_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve edges with optional filtering"""
        start_time = time.time()
        
        edges = []
        
        for src, tgt, key in self.graph.edges(keys=True):
            # Apply filters
            if source and src != source:
                continue
            if target and tgt != target:
                continue
            if relationship_type and key != relationship_type:
                continue
            
            # Get edge properties
            edge_key = f"{src}->{tgt}:{key}"
            properties = self.edge_properties.get(edge_key, {})
            
            edges.append({
                'source': src,
                'target': tgt,
                'relationship_type': key,
                'properties': properties.copy()
            })
        
        self.stats['queries_executed'] += 1
        self.stats['last_operation_time'] = time.time() - start_time
        return edges
    
    def get_neighbors(self, node_id: str, direction: str = 'both') -> List[str]:
        """Get neighboring nodes"""
        start_time = time.time()
        
        neighbors = set()
        
        if direction in ['outgoing', 'both']:
            neighbors.update(self.graph.successors(node_id))
        
        if direction in ['incoming', 'both']:
            neighbors.update(self.graph.predecessors(node_id))
        
        self.stats['queries_executed'] += 1
        self.stats['last_operation_time'] = time.time() - start_time
        return list(neighbors)
    
    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists"""
        return node_id in self.graph.nodes()
    
    def edge_exists(self, source: str, target: str, relationship_type: str = None) -> bool:
        """Check if an edge exists"""
        if not self.graph.has_edge(source, target):
            return False
        
        if relationship_type is None:
            return True
        
        # Check specific relationship type
        edge_data = self.graph.get_edge_data(source, target)
        return relationship_type in edge_data.keys()
    
    def get_node_count(self) -> int:
        """Get total number of nodes"""
        return self.graph.number_of_nodes()
    
    def get_edge_count(self) -> int:
        """Get total number of edges"""
        return self.graph.number_of_edges()
    
    def update_node_properties(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties"""
        if node_id not in self.node_properties:
            return False
        
        self.node_properties[node_id].update(properties)
        return True
    
    def update_edge_properties(self, source: str, target: str, relationship_type: str, properties: Dict[str, Any]) -> bool:
        """Update edge properties"""
        edge_key = f"{source}->{target}:{relationship_type}"
        
        if edge_key not in self.edge_properties:
            return False
        
        self.edge_properties[edge_key].update(properties)
        return True
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges"""
        if not self.node_exists(node_id):
            return False
        
        # Remove from NetworkX graph
        self.graph.remove_node(node_id)
        
        # Clean up properties
        if node_id in self.node_properties:
            del self.node_properties[node_id]
        
        # Clean up edge properties
        keys_to_remove = []
        for edge_key in self.edge_properties:
            if node_id in edge_key:
                keys_to_remove.append(edge_key)
        
        for key in keys_to_remove:
            del self.edge_properties[key]
        
        return True
    
    def delete_edge(self, source: str, target: str, relationship_type: str = None) -> bool:
        """Delete an edge"""
        if not self.edge_exists(source, target, relationship_type):
            return False
        
        if relationship_type:
            # Remove specific edge
            self.graph.remove_edge(source, target, key=relationship_type)
            edge_key = f"{source}->{target}:{relationship_type}"
            if edge_key in self.edge_properties:
                del self.edge_properties[edge_key]
        else:
            # Remove all edges between nodes
            self.graph.remove_edge(source, target)
            keys_to_remove = [k for k in self.edge_properties if k.startswith(f"{source}->{target}:")]
            for key in keys_to_remove:
                del self.edge_properties[key]
        
        return True
    
    def clear_graph(self) -> bool:
        """Clear all nodes and edges"""
        self.graph.clear()
        self.node_properties.clear()
        self.edge_properties.clear()
        return True
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom query (simplified for memory adapter)"""
        # For memory adapter, implement basic query patterns
        query = query.lower().strip()
        
        if query.startswith('match'):
            # Simple pattern matching
            if 'return count' in query:
                if 'nodes' in query or 'entity' in query:
                    return [{'count': self.get_node_count()}]
                elif 'edges' in query or 'relationship' in query:
                    return [{'count': self.get_edge_count()}]
        
        # Default: return empty result
        return []
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        relationship_types = set()
        
        for _, _, key in self.graph.edges(keys=True):
            relationship_types.add(key)
        
        return {
            'nodes': self.get_node_count(),
            'edges': self.get_edge_count(),
            'relationship_types': len(relationship_types),
            'relationship_type_list': list(relationship_types),
            'performance': self.stats,
            'backend': 'memory',
            'connected': self.connected
        }
    
    def export_graph(self, format: str = 'json') -> str:
        """Export graph data"""
        if format != 'json':
            raise ValueError("Only JSON format supported for memory adapter")
        
        # Export nodes
        nodes = []
        for node_id in self.graph.nodes():
            properties = self.node_properties.get(node_id, {})
            nodes.append({
                'id': node_id,
                'properties': properties
            })
        
        # Export edges
        edges = []
        for source, target, key in self.graph.edges(keys=True):
            edge_key = f"{source}->{target}:{key}"
            properties = self.edge_properties.get(edge_key, {})
            edges.append({
                'source': source,
                'target': target,
                'relationship_type': key,
                'properties': properties
            })
        
        export_data = {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'export_time': time.time(),
                'backend': 'memory',
                'stats': self.get_graph_stats()
            }
        }
        
        return json.dumps(export_data, indent=2)
    
    def import_graph(self, data: str, format: str = 'json') -> bool:
        """Import graph data"""
        if format != 'json':
            raise ValueError("Only JSON format supported for memory adapter")
        
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
    
    def get_networkx_graph(self) -> nx.MultiDiGraph:
        """Get the underlying NetworkX graph for advanced operations"""
        return self.graph
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'connection_status': self.connected,
            'graph_stats': self.get_graph_stats(),
            'operation_stats': self.stats,
            'memory_usage': {
                'nodes': len(self.node_properties),
                'edges': len(self.edge_properties),
                'nx_nodes': self.graph.number_of_nodes(),
                'nx_edges': self.graph.number_of_edges()
            }
        }
