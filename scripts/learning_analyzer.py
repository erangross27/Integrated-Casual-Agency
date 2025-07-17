#!/usr/bin/env python3
"""
TRUE AGI Learning Analyzer
Analyzes what the AGI has learned and discovered about physics and the world
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ica_framework.sandbox.agi_agent import AGIAgent
from ica_framework.sandbox.world_simulator import WorldSimulator
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph


class LearningAnalyzer:
    """Analyzes AGI learning progress and discovered knowledge"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_physics_learning(self, world_simulator: WorldSimulator, agi_agent: AGIAgent):
        """Analyze what physics concepts the AGI has learned"""
        print("🔬 ANALYZING AGI PHYSICS LEARNING")
        print("=" * 50)
        
        # Get world simulation statistics
        world_stats = world_simulator.get_learning_statistics()
        sim_stats = world_stats.get('simulation', {})
        learning_stats = world_stats.get('learning', {})
        
        print(f"📊 Simulation Statistics:")
        print(f"   • Total Steps: {sim_stats.get('steps', 0):,}")
        print(f"   • Simulation Speed: {sim_stats.get('steps_per_second', 0):.1f} steps/sec")
        print(f"   • Discovery Events: {len(world_simulator.discovery_events)}")
        print(f"   • Curiosity Events: {len(world_simulator.curiosity_events)}")
        
        # Analyze discovery patterns
        self._analyze_discovery_patterns(world_simulator)
        
        # Analyze agent knowledge
        self._analyze_agent_knowledge(agi_agent)
        
        # Analyze learning progression
        self._analyze_learning_progression(agi_agent)
        
        return self.analysis_results
    
    def _analyze_discovery_patterns(self, world_simulator):
        """Analyze patterns in AGI discoveries"""
        print(f"\n🧠 DISCOVERY PATTERN ANALYSIS")
        print("-" * 30)
        
        # Categorize discoveries
        discovery_types = {}
        for event in world_simulator.discovery_events[-50:]:  # Last 50 discoveries
            pattern_type = event.get('pattern', 'unknown')
            if pattern_type not in discovery_types:
                discovery_types[pattern_type] = []
            discovery_types[pattern_type].append(event)
        
        for pattern_type, events in discovery_types.items():
            print(f"   • {pattern_type}: {len(events)} discoveries")
            if events:
                latest = events[-1]
                print(f"     Latest: Step {latest.get('step', 0)} - {latest.get('context', {}).get('description', 'No description')}")
        
        # Analyze curiosity triggers
        print(f"\n🤔 CURIOSITY TRIGGERS ANALYSIS")
        print("-" * 30)
        
        curiosity_types = {}
        for event in world_simulator.curiosity_events[-20:]:  # Last 20 curiosity events
            trigger_type = event.get('trigger', 'unknown')
            if trigger_type not in curiosity_types:
                curiosity_types[trigger_type] = []
            curiosity_types[trigger_type].append(event)
        
        for trigger_type, events in curiosity_types.items():
            print(f"   • {trigger_type}: {len(events)} triggers")
    
    def _analyze_agent_knowledge(self, agi_agent):
        """Analyze the AGI agent's accumulated knowledge"""
        print(f"\n🧠 AGI KNOWLEDGE BASE ANALYSIS")
        print("-" * 30)
        
        # Learning progress metrics
        progress = agi_agent.learning_progress
        print(f"📈 Learning Metrics:")
        print(f"   • Concepts Learned: {progress.get('concepts_learned', 0):,}")
        print(f"   • Hypotheses Formed: {progress.get('hypotheses_formed', 0)}")
        print(f"   • Hypotheses Confirmed: {progress.get('hypotheses_confirmed', 0)}")
        print(f"   • Causal Relationships: {progress.get('causal_relationships_discovered', 0)}")
        print(f"   • Pattern Recognition: {progress.get('patterns_recognized', 0):,}")
        
        # Memory analysis
        print(f"\n💾 Memory Systems:")
        print(f"   • Short-term Memory: {len(agi_agent.short_term_memory)} items")
        print(f"   • Long-term Memory: {len(agi_agent.long_term_memory)} items")
        print(f"   • Episodic Memory: {len(agi_agent.episodic_memory)} episodes")
        
        # Curiosity and attention
        print(f"\n🎯 Cognitive State:")
        print(f"   • Curiosity Level: {agi_agent.curiosity_level:.2f}")
        print(f"   • Exploration Rate: {agi_agent.exploration_rate:.2f}")
        print(f"   • Attention Focus: {agi_agent.attention_focus or 'None'}")
        
        # Knowledge base content
        print(f"\n📚 Knowledge Base Content:")
        print(f"   • Total Concepts: {len(agi_agent.knowledge_base)}")
        print(f"   • Causal Models: {len(agi_agent.causal_models)}")
        
        # Show recent knowledge if available
        if agi_agent.knowledge_base:
            print(f"\n🔍 Recent Knowledge Samples:")
            recent_concepts = list(agi_agent.knowledge_base.items())[-5:]
            for concept, data in recent_concepts:
                print(f"   • {concept}: {str(data)[:100]}...")
    
    def _analyze_learning_progression(self, agi_agent):
        """Analyze how learning has progressed over time"""
        print(f"\n📈 LEARNING PROGRESSION ANALYSIS")
        print("-" * 30)
        
        # Active hypotheses
        print(f"🧪 Active Research:")
        print(f"   • Active Hypotheses: {len(agi_agent.active_hypotheses)}")
        print(f"   • Tested Hypotheses: {len(agi_agent.tested_hypotheses)}")
        
        if agi_agent.active_hypotheses:
            print(f"\n🔬 Current Active Hypotheses:")
            for i, hypothesis in enumerate(agi_agent.active_hypotheses[-3:], 1):
                print(f"   {i}. Type: {hypothesis.get('type', 'unknown')}")
                print(f"      Statement: {hypothesis.get('statement', 'No statement')}")
                print(f"      Testable: {hypothesis.get('testable', False)}")
        
        # Recent discoveries in memory
        if agi_agent.episodic_memory:
            print(f"\n💭 Recent Learning Episodes:")
            recent_episodes = agi_agent.episodic_memory[-3:]
            for i, episode in enumerate(recent_episodes, 1):
                print(f"   {i}. {episode.get('type', 'unknown')} - {episode.get('description', 'No description')[:80]}...")
    
    def save_analysis_report(self, filename="agi_learning_analysis.json"):
        """Save analysis results to file"""
        report_path = Path(filename)
        with open(report_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        print(f"\n📄 Analysis report saved to: {report_path.absolute()}")


def main():
    """Main analysis function"""
    print("🧠 TRUE AGI LEARNING ANALYZER")
    print("=" * 50)
    print("This tool analyzes what your AGI has learned about physics and the world\n")
    
    # Initialize knowledge graph
    try:
        knowledge_graph = EnhancedKnowledgeGraph(backend='memory')
        if not knowledge_graph.connect():
            print("⚠️ Warning: Could not connect to knowledge graph")
            knowledge_graph = None
    except Exception as e:
        print(f"⚠️ Warning: Knowledge graph error: {e}")
        knowledge_graph = None
    
    # Initialize world simulator and AGI agent
    print("🌍 Initializing world simulator...")
    world_simulator = WorldSimulator()
    
    print("🤖 Initializing AGI agent...")
    agi_agent = AGIAgent(world_simulator, knowledge_graph)
    
    # Load any existing learning state
    print("📊 Loading existing learning data...")
    
    # Analyze learning
    analyzer = LearningAnalyzer()
    results = analyzer.analyze_physics_learning(world_simulator, agi_agent)
    
    # Save analysis report
    analyzer.save_analysis_report()
    
    print(f"\n✅ Analysis complete!")
    print(f"🎯 The AGI has discovered {agi_agent.learning_progress.get('concepts_learned', 0)} concepts")
    print(f"🧪 Currently testing {len(agi_agent.active_hypotheses)} hypotheses")
    print(f"🔬 Confirmed {agi_agent.learning_progress.get('hypotheses_confirmed', 0)} scientific relationships")


if __name__ == "__main__":
    main()
