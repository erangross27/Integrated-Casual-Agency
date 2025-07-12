"""
Visualization utilities for ICA Framework
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Optional imports with fallbacks
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    px = None
    make_subplots = None

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    TSNE = None
    PCA = None


class Visualizer:
    """Visualization tools for ICA Framework analysis"""
    
    def __init__(self, style: str = "default"):
        self.has_seaborn = HAS_SEABORN
        self.has_plotly = HAS_PLOTLY
        self.has_sklearn = HAS_SKLEARN
        
        # Set matplotlib style
        if self.has_seaborn and style == "seaborn":
            plt.style.use("seaborn")
            self.color_palette = sns.color_palette("husl", 10)
        else:
            plt.style.use("default")
            # Create a simple color palette if seaborn is not available
            self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_graph(self, graph: nx.Graph, 
                   node_colors: Optional[Dict] = None,
                   edge_colors: Optional[Dict] = None,
                   title: str = "Knowledge Graph",
                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot networkx graph with customizable styling"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate layout
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Draw nodes
        if node_colors:
            node_color_list = [node_colors.get(node, 'lightblue') for node in graph.nodes()]
        else:
            node_color_list = 'lightblue'
            
        nx.draw_networkx_nodes(graph, pos, node_color=node_color_list, 
                              node_size=300, alpha=0.7, ax=ax)
        
        # Draw edges
        if edge_colors:
            edges = graph.edges()
            edge_color_list = [edge_colors.get(edge, 'gray') for edge in edges]
        else:
            edge_color_list = 'gray'
            
        nx.draw_networkx_edges(graph, pos, edge_color=edge_color_list, 
                              alpha=0.5, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
        
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        return fig
    
    def plot_embedding_space(self, embeddings: np.ndarray, 
                           labels: Optional[np.ndarray] = None,
                           method: str = "pca",
                           title: str = "Embedding Space",
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot 2D representation of embeddings"""
        
        if not self.has_sklearn:
            # Simple fallback - plot first two dimensions
            if embeddings.shape[1] >= 2:
                embeddings_2d = embeddings[:, :2]
            else:
                # Pad with zeros if less than 2 dimensions
                embeddings_2d = np.column_stack([
                    embeddings[:, 0] if embeddings.shape[1] > 0 else np.zeros(embeddings.shape[0]),
                    embeddings[:, 1] if embeddings.shape[1] > 1 else np.zeros(embeddings.shape[0])
                ])
            method = "first_two_dims"
        else:
            # Reduce dimensionality using sklearn
            if method == "tsne":
                reducer = TSNE(n_components=2, random_state=42)
            elif method == "pca":
                reducer = PCA(n_components=2)
            else:
                # Default to PCA if unknown method
                reducer = PCA(n_components=2)
                method = "pca"
            
            embeddings_2d = reducer.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is not None:
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        ax.set_title(f"{title} ({method.upper()})", fontsize=16)
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        
        return fig
    
    def plot_metrics_history(self, metrics_history: Dict[str, List[Dict]],
                           title: str = "Metrics Over Time"):
        """Plot metrics history using plotly or matplotlib"""
        if not self.has_plotly:
            # Fallback to matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for metric_name, history in metrics_history.items():
                steps = [item["step"] for item in history]
                values = [item["value"] for item in history]
                ax.plot(steps, values, marker='o', label=metric_name)
            
            ax.set_title(title)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
            return fig
        
        # Use plotly if available
        fig = go.Figure()
        
        for metric_name, history in metrics_history.items():
            steps = [item["step"] for item in history]
            values = [item["value"] for item in history]
            
            fig.add_trace(go.Scatter(
                x=steps,
                y=values,
                mode='lines+markers',
                name=metric_name,
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Step",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_utility_scores(self, utility_scores: Dict[str, List[float]],
                          title: str = "Utility Score Evolution"):
        """Plot utility score evolution for different concepts"""
        if not self.has_plotly:
            # Fallback to matplotlib
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for concept_name, scores in utility_scores.items():
                ax.plot(scores, label=concept_name, marker='o')
            
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Utility Score")
            ax.legend()
            ax.grid(True)
            return fig
        
        # Use plotly if available
        fig = go.Figure()
        
        for concept_name, scores in utility_scores.items():
            fig.add_trace(go.Scatter(
                y=scores,
                mode='lines',
                name=concept_name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Step",
            yaxis_title="Utility Score",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_confidence_heatmap(self, confidence_matrix: np.ndarray,
                              labels: Optional[List[str]] = None,
                              title: str = "Confidence Heatmap",
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot confidence heatmap"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if self.has_seaborn:
            # Use seaborn if available
            sns.heatmap(confidence_matrix, 
                       annot=True, 
                       fmt='.2f', 
                       cmap='YlOrRd',
                       xticklabels=labels,
                       yticklabels=labels,
                       ax=ax)
        else:
            # Fallback to matplotlib
            im = ax.imshow(confidence_matrix, cmap='YlOrRd', aspect='auto')
            
            # Add text annotations
            for i in range(confidence_matrix.shape[0]):
                for j in range(confidence_matrix.shape[1]):
                    text = ax.text(j, i, f'{confidence_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black")
            
            # Set labels if provided
            if labels:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        ax.set_title(title, fontsize=16)
        
        return fig
    
    def plot_motif_discovery_results(self, discovered_motifs: List[Any],
                                   ground_truth_motifs: List[Any],
                                   metrics: Dict[str, float],
                                   title: str = "Motif Discovery Results"):
        """Plot motif discovery results"""
        
        if not self.has_plotly:
            # Fallback to matplotlib
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Metrics bar chart
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            ax3.bar(metric_names, metric_values)
            ax3.set_title('Precision/Recall/F1')
            ax3.set_ylabel('Score')
            
            # Motif size distribution
            discovered_sizes = [len(str(motif)) for motif in discovered_motifs]
            truth_sizes = [len(str(motif)) for motif in ground_truth_motifs]
            
            ax4.hist(discovered_sizes, alpha=0.7, label='Discovered', bins=10)
            ax4.hist(truth_sizes, alpha=0.7, label='Ground Truth', bins=10)
            ax4.set_title('Motif Size Distribution')
            ax4.set_xlabel('Size')
            ax4.set_ylabel('Count')
            ax4.legend()
            
            # Simple motif count visualization
            ax1.bar(['Discovered', 'Ground Truth'], 
                   [len(discovered_motifs), len(ground_truth_motifs)])
            ax1.set_title('Motif Count Comparison')
            ax1.set_ylabel('Count')
            
            # Hide the unused subplot
            ax2.axis('off')
            ax2.text(0.5, 0.5, 'Motif Visualization\n(Interactive version\navailable with Plotly)', 
                    ha='center', va='center', transform=ax2.transAxes)
            
            plt.suptitle(title)
            plt.tight_layout()
            return fig
        
        # Create subplots using plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Discovered Motifs', 'Ground Truth Motifs', 
                          'Precision/Recall/F1', 'Motif Size Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Metrics bar chart
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name="Metrics"),
            row=2, col=1
        )
        
        # Motif size distribution
        discovered_sizes = [len(str(motif)) for motif in discovered_motifs]
        truth_sizes = [len(str(motif)) for motif in ground_truth_motifs]
        
        fig.add_trace(
            go.Histogram(x=discovered_sizes, name="Discovered", opacity=0.7),
            row=2, col=2
        )
        fig.add_trace(
            go.Histogram(x=truth_sizes, name="Ground Truth", opacity=0.7),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=title,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def plot_ablation_study(self, baseline_results: Dict[str, float],
                          enhanced_results: Dict[str, float],
                          title: str = "Ablation Study Results"):
        """Plot ablation study comparison"""
        
        if not self.has_plotly:
            # Fallback to matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = list(baseline_results.keys())
            baseline_values = list(baseline_results.values())
            enhanced_values = list(enhanced_results.values())
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, baseline_values, width, label='Baseline', color='lightcoral')
            ax.bar(x + width/2, enhanced_values, width, label='Enhanced', color='lightblue')
            
            ax.set_title(title)
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
        
        # Use plotly if available
        metrics = list(baseline_results.keys())
        baseline_values = list(baseline_results.values())
        enhanced_values = list(enhanced_results.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=baseline_values,
            name='Baseline',
            marker_color='lightcoral'
        ))
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=enhanced_values,
            name='Enhanced',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group',
            template='plotly_white'
        )
        
        return fig
    
    def create_interactive_graph(self, graph: nx.Graph,
                               node_attributes: Optional[Dict] = None,
                               edge_attributes: Optional[Dict] = None,
                               title: str = "Interactive Knowledge Graph"):
        """Create interactive graph visualization"""
        
        if not self.has_plotly:
            # Fallback to matplotlib
            return self.plot_graph(graph, title=title)
        
        # Use plotly for interactive visualization
        pos = nx.spring_layout(graph, k=1, iterations=50)
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_text = [str(node) for node in graph.nodes()]
        
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='gray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=10,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive graph visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
