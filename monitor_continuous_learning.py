#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for Continuous Learning

This script provides real-time visualization and monitoring of the continuous learning agent.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
from collections import deque

try:
    import matplotlib.animation as animation
    from matplotlib.widgets import Button
    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False
    print("Warning: matplotlib animation not available. Using static plots.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not available. Using matplotlib only.")


class ContinuousLearningMonitor:
    """
    Real-time monitoring dashboard for the continuous learning agent.
    """
    
    def __init__(self, data_dir: str = "continuous_learning_data", update_interval: int = 5):
        self.data_dir = Path(data_dir)
        self.update_interval = update_interval
        
        # Data storage
        self.metrics_history = deque(maxlen=1000)
        self.recent_performance = deque(maxlen=100)
        self.graph_growth = deque(maxlen=500)
        self.confidence_trend = deque(maxlen=500)
        
        # Monitoring state
        self.is_monitoring = False
        self.last_update = datetime.now()
        
        # Visualization setup
        if HAS_ANIMATION:
            self.setup_matplotlib_dashboard()
        
        print(f"Monitor initialized for directory: {data_dir}")
    
    def setup_matplotlib_dashboard(self):
        """Setup the matplotlib-based real-time dashboard."""
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('ICA Framework - Continuous Learning Monitor', fontsize=16)
        
        # Configure subplots
        self.axes[0, 0].set_title('Confidence Trend')
        self.axes[0, 0].set_xlabel('Time Steps')
        self.axes[0, 0].set_ylabel('Confidence')
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].set_title('Knowledge Graph Growth')
        self.axes[0, 1].set_xlabel('Time Steps')
        self.axes[0, 1].set_ylabel('Number of Nodes')
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].set_title('Learning Rate')
        self.axes[1, 0].set_xlabel('Time Steps')
        self.axes[1, 0].set_ylabel('Experiments per Step')
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title('Environment Complexity')
        self.axes[1, 1].set_xlabel('Time Steps')
        self.axes[1, 1].set_ylabel('Complexity Factor')
        self.axes[1, 1].grid(True)
        
        # Initialize empty lines
        self.lines = {}
        self.lines['confidence'], = self.axes[0, 0].plot([], [], 'b-', linewidth=2)
        self.lines['graph_size'], = self.axes[0, 1].plot([], [], 'g-', linewidth=2)
        self.lines['learning_rate'], = self.axes[1, 0].plot([], [], 'r-', linewidth=2)
        self.lines['complexity'], = self.axes[1, 1].plot([], [], 'm-', linewidth=2)
        
        plt.tight_layout()
    
    def load_latest_data(self) -> Optional[Dict[str, Any]]:
        """Load the most recent monitoring data."""
        try:
            # Find latest performance metrics file
            metric_files = list(self.data_dir.glob("performance_metrics_*.json"))
            if not metric_files:
                return None
            
            latest_file = max(metric_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                metrics = json.load(f)
            
            # Find latest learning history file
            history_files = list(self.data_dir.glob("learning_history_*.json"))
            if history_files:
                latest_history_file = max(history_files, key=lambda x: x.stat().st_mtime)
                with open(latest_history_file, 'r') as f:
                    history = json.load(f)
                    if history:
                        latest_record = history[-1]
                        metrics['latest_step'] = latest_record
            
            return metrics
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def update_data(self):
        """Update monitoring data from latest files."""
        data = self.load_latest_data()
        if not data:
            return
        
        # Update confidence trend
        if 'confidence_trend' in data and data['confidence_trend']:
            self.confidence_trend.extend(data['confidence_trend'][-100:])
        
        # Update graph growth
        if 'knowledge_graph_size' in data and data['knowledge_graph_size']:
            self.graph_growth.extend(data['knowledge_graph_size'][-100:])
        
        # Calculate learning rate from recent performance
        if 'latest_step' in data:
            step_data = data['latest_step']
            learning_rate = step_data.get('results', {}).get('experiments_conducted', 0)
            self.recent_performance.append({
                'step': step_data.get('step', 0),
                'learning_rate': learning_rate,
                'complexity': step_data.get('environment_complexity', 1.0),
                'step_time': step_data.get('step_time', 0.0)
            })
        
        self.last_update = datetime.now()
    
    def update_matplotlib_plots(self, frame):
        """Update matplotlib plots with new data."""
        if not self.is_monitoring:
            return self.lines.values()
        
        self.update_data()
        
        # Update confidence plot
        if self.confidence_trend:
            x_conf = list(range(len(self.confidence_trend)))
            y_conf = list(self.confidence_trend)
            self.lines['confidence'].set_data(x_conf, y_conf)
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()
        
        # Update graph growth plot
        if self.graph_growth:
            x_graph = list(range(len(self.graph_growth)))
            y_graph = list(self.graph_growth)
            self.lines['graph_size'].set_data(x_graph, y_graph)
            self.axes[0, 1].relim()
            self.axes[0, 1].autoscale_view()
        
        # Update learning rate plot
        if self.recent_performance:
            x_lr = [p['step'] for p in self.recent_performance]
            y_lr = [p['learning_rate'] for p in self.recent_performance]
            self.lines['learning_rate'].set_data(x_lr, y_lr)
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()
        
        # Update complexity plot
        if self.recent_performance:
            x_comp = [p['step'] for p in self.recent_performance]
            y_comp = [p['complexity'] for p in self.recent_performance]
            self.lines['complexity'].set_data(x_comp, y_comp)
            self.axes[1, 1].relim()
            self.axes[1, 1].autoscale_view()
        
        return self.lines.values()
    
    def start_monitoring(self):
        """Start the real-time monitoring dashboard."""
        self.is_monitoring = True
        
        if HAS_ANIMATION:
            # Start matplotlib animation
            self.ani = animation.FuncAnimation(
                self.fig, self.update_matplotlib_plots,
                interval=self.update_interval * 1000,
                blit=False, cache_frame_data=False
            )
            plt.show()
        else:
            # Fallback to periodic updates
            self.text_monitoring()
    
    def text_monitoring(self):
        """Text-based monitoring when animation is not available."""
        print("Starting text-based monitoring...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while self.is_monitoring:
                self.update_data()
                self.print_status()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    
    def print_status(self):
        """Print current status to console."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🧠 ICA Framework - Continuous Learning Monitor")
        print("=" * 60)
        print(f"Last Update: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        if self.confidence_trend:
            current_confidence = self.confidence_trend[-1]
            avg_confidence = np.mean(list(self.confidence_trend))
            print(f"📊 Current Confidence: {current_confidence:.3f}")
            print(f"📈 Average Confidence: {avg_confidence:.3f}")
        
        if self.graph_growth:
            current_nodes = self.graph_growth[-1]
            growth = current_nodes - self.graph_growth[0] if len(self.graph_growth) > 1 else 0
            print(f"🕸️  Graph Nodes: {current_nodes}")
            print(f"📈 Graph Growth: +{growth} nodes")
        
        if self.recent_performance:
            latest = self.recent_performance[-1]
            print(f"🔬 Current Step: {latest['step']}")
            print(f"⚡ Learning Rate: {latest['learning_rate']} experiments/step")
            print(f"🌍 Environment Complexity: {latest['complexity']:.3f}")
            print(f"⏱️  Step Time: {latest['step_time']:.3f}s")
        
        print()
        print("Press Ctrl+C to stop monitoring")
        print("-" * 60)
    
    def generate_plotly_dashboard(self, output_file: str = "monitoring_dashboard.html"):
        """Generate an interactive Plotly dashboard."""
        if not HAS_PLOTLY:
            print("Plotly not available. Install with: pip install plotly")
            return
        
        data = self.load_latest_data()
        if not data:
            print("No data available for dashboard generation.")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confidence Trend', 'Knowledge Graph Growth',
                          'Learning Rate', 'Environment Complexity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Confidence trend
        if 'confidence_trend' in data:
            fig.add_trace(
                go.Scatter(y=data['confidence_trend'], mode='lines', name='Confidence'),
                row=1, col=1
            )
        
        # Graph growth
        if 'knowledge_graph_size' in data:
            fig.add_trace(
                go.Scatter(y=data['knowledge_graph_size'], mode='lines', name='Graph Size'),
                row=1, col=2
            )
        
        # Add more traces for other metrics...
        
        fig.update_layout(
            title="ICA Framework - Continuous Learning Dashboard",
            height=800,
            showlegend=False
        )
        
        fig.write_html(output_file)
        print(f"Interactive dashboard saved to: {output_file}")
    
    def stop_monitoring(self):
        """Stop the monitoring."""
        self.is_monitoring = False


def main():
    """Main entry point for the monitoring dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Learning Monitor")
    parser.add_argument("--data-dir", type=str, default="continuous_learning_data",
                       help="Directory containing learning data")
    parser.add_argument("--update-interval", type=int, default=5,
                       help="Update interval in seconds")
    parser.add_argument("--generate-dashboard", action="store_true",
                       help="Generate static HTML dashboard")
    parser.add_argument("--text-only", action="store_true",
                       help="Use text-only monitoring")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = ContinuousLearningMonitor(args.data_dir, args.update_interval)
    
    if args.generate_dashboard:
        monitor.generate_plotly_dashboard()
        return
    
    if args.text_only:
        monitor.text_monitoring()
    else:
        monitor.start_monitoring()


if __name__ == "__main__":
    main()
