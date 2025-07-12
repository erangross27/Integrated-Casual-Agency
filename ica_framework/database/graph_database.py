"""
Abstract base class for graph database adapters
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import json


class GraphDatabase(ABC):
    """Abstract base class for graph database implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close database connection"""
        pass
    
    @abstractmethod
    def add_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Add a node to the graph"""
        pass
    
    @abstractmethod
    def add_edge(self, source: str, target: str, relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        """Add an edge between two nodes"""
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by ID"""
        pass
    
    @abstractmethod
    def get_edges(self, source: str = None, target: str = None, relationship_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve edges with optional filtering"""
        pass
    
    @abstractmethod
    def get_neighbors(self, node_id: str, direction: str = 'both') -> List[str]:
        """Get neighboring nodes"""
        pass
    
    @abstractmethod
    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists"""
        pass
    
    @abstractmethod
    def edge_exists(self, source: str, target: str, relationship_type: str = None) -> bool:
        """Check if an edge exists"""
        pass
    
    @abstractmethod
    def get_node_count(self) -> int:
        """Get total number of nodes"""
        pass
    
    @abstractmethod
    def get_edge_count(self) -> int:
        """Get total number of edges"""
        pass
    
    @abstractmethod
    def update_node_properties(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties"""
        pass
    
    @abstractmethod
    def update_edge_properties(self, source: str, target: str, relationship_type: str, properties: Dict[str, Any]) -> bool:
        """Update edge properties"""
        pass
    
    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges"""
        pass
    
    @abstractmethod
    def delete_edge(self, source: str, target: str, relationship_type: str = None) -> bool:
        """Delete an edge"""
        pass
    
    @abstractmethod
    def clear_graph(self) -> bool:
        """Clear all nodes and edges"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom query"""
        pass
    
    @abstractmethod
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        pass
    
    @abstractmethod
    def export_graph(self, format: str = 'json') -> str:
        """Export graph data"""
        pass
    
    @abstractmethod
    def import_graph(self, data: str, format: str = 'json') -> bool:
        """Import graph data"""
        pass
    
    def backup_graph(self, backup_path: str) -> bool:
        """Create a backup of the graph"""
        try:
            graph_data = self.export_graph('json')
            with open(backup_path, 'w') as f:
                f.write(graph_data)
            return True
        except Exception as e:
            print(f"Backup failed: {e}")
            return False
    
    def restore_graph(self, backup_path: str) -> bool:
        """Restore graph from backup"""
        try:
            with open(backup_path, 'r') as f:
                graph_data = f.read()
            return self.import_graph(graph_data, 'json')
        except Exception as e:
            print(f"Restore failed: {e}")
            return False
