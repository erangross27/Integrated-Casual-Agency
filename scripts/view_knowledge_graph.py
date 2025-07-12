#!/usr/bin/env python3
"""
Knowledge Graph Viewer

View the current state of the agent's causal knowledge graph.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

def load_latest_agent_state(data_dir="data/continuous_learning_data"):
    """Load the most recent agent state."""
    data_path = Path(data_dir)
    
    # Find latest agent state file
    state_files = list(data_path.glob("agent_state_*.json"))
    if not state_files:
        print(f"No agent state files found in {data_dir}")
        return None
    
    latest_file = max(state_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def visualize_knowledge_graph(agent_state, save_plot=True):
    """Visualize the causal knowledge graph."""
    if not agent_state or 'knowledge_graph' not in agent_state:
        print("No knowledge graph found in agent state")
        return
    
    kg_data = agent_state['knowledge_graph']
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node_data in kg_data.get('nodes', {}).items():
        G.add_node(node_id, **node_data)
    
    # Add edges with confidence
    edges_info = []
    for edge_id, edge_data in kg_data.get('edges', {}).items():
        source = edge_data.get('source')
        target = edge_data.get('target')
        confidence = edge_data.get('confidence', 0.5)
        relation = edge_data.get('relation', 'unknown')
        
        if source and target:
            G.add_edge(source, target, 
                      confidence=confidence, 
                      relation=relation,
                      edge_id=edge_id)
            edges_info.append((source, target, confidence, relation))
    
    print(f"\n🧠 Knowledge Graph Statistics:")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Density: {nx.density(G):.3f}")
    
    if G.number_of_nodes() == 0:
        print("Graph is empty - agent may not have learned anything yet.")
        return
    
    # Calculate layout
    if G.number_of_nodes() > 50:
        # Use spring layout for large graphs
        pos = nx.spring_layout(G, k=1, iterations=50)
    else:
        # Use circular layout for smaller graphs
        pos = nx.circular_layout(G)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Draw nodes
    node_sizes = [300 + len(node_id) * 20 for node_id in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7)
    
    # Draw edges with thickness based on confidence
    if edges_info:
        confidences = [conf for _, _, conf, _ in edges_info]
        max_conf = max(confidences) if confidences else 1.0
        min_conf = min(confidences) if confidences else 0.0
        
        for source, target, confidence, relation in edges_info:
            # Edge thickness based on confidence
            width = 0.5 + (confidence / max_conf) * 3
            alpha = 0.3 + (confidence / max_conf) * 0.7
            
            nx.draw_networkx_edges(G, pos, [(source, target)], 
                                 width=width, alpha=alpha, 
                                 edge_color='gray', arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Add title with stats
    plt.title(f"ICA Agent Knowledge Graph\n"
              f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, "
              f"Density: {nx.density(G):.3f}", fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"knowledge_graph_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Graph saved as: {filename}")
    
    plt.show()

def print_graph_details(agent_state):
    """Print detailed information about the knowledge graph."""
    if not agent_state or 'knowledge_graph' not in agent_state:
        return
    
    kg_data = agent_state['knowledge_graph']
    
    print(f"\n📋 Detailed Knowledge Graph Analysis:")
    print(f"   Total Nodes: {len(kg_data.get('nodes', {}))}")
    print(f"   Total Edges: {len(kg_data.get('edges', {}))}")
    
    # Analyze confidence distribution
    confidences = []
    relations = {}
    
    for edge_data in kg_data.get('edges', {}).values():
        conf = edge_data.get('confidence', 0.5)
        confidences.append(conf)
        
        rel = edge_data.get('relation', 'unknown')
        relations[rel] = relations.get(rel, 0) + 1
    
    if confidences:
        import numpy as np
        print(f"\n🎯 Confidence Statistics:")
        print(f"   Average: {np.mean(confidences):.3f}")
        print(f"   Std Dev: {np.std(confidences):.3f}")
        print(f"   Min: {np.min(confidences):.3f}")
        print(f"   Max: {np.max(confidences):.3f}")
    
    if relations:
        print(f"\n🔗 Relation Types:")
        for rel, count in sorted(relations.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {rel}: {count}")

def main():
    parser = argparse.ArgumentParser(description="View ICA Agent Knowledge Graph")
    parser.add_argument("--data-dir", default="data/continuous_learning_data",
                       help="Directory containing agent data")
    parser.add_argument("--no-plot", action="store_true",
                       help="Don't show visual plot")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed analysis")
    
    args = parser.parse_args()
    
    # Load agent state
    agent_state = load_latest_agent_state(args.data_dir)
    if not agent_state:
        return
    
    print(f"\n🤖 Agent State Summary:")
    print(f"   Global Confidence: {agent_state.get('global_confidence', 0):.3f}")
    print(f"   Total Steps: {agent_state.get('step_count', 0)}")
    
    # Show detailed analysis
    if args.details:
        print_graph_details(agent_state)
    
    # Visualize graph
    if not args.no_plot:
        try:
            visualize_knowledge_graph(agent_state)
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Try running with --no-plot flag")

if __name__ == "__main__":
    main()
