#!/usr/bin/env python3
"""
AGI Intelligence Sophistication Analyzer
Analyzes the current knowledge graph to assess intelligence level
"""

import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

def analyze_intelligence_level():
    """Analyze current AGI intelligence sophistication"""
    
    # Load Neo4j config
    with open('config/database/neo4j.json', 'r') as f:
        config_data = json.load(f)
        neo4j_config = config_data['config']
    
    # Connect to knowledge graph
    kg = EnhancedKnowledgeGraph(backend='neo4j', config=neo4j_config)
    
    print("🧠 AGI Intelligence Sophistication Analysis")
    print("=" * 60)
    
    # 1. Basic Knowledge Scale
    stats = kg.get_stats()
    entities = stats['nodes']
    relationships = stats['edges']
    density = relationships / entities if entities > 0 else 0
    
    print(f"📊 Knowledge Scale:")
    print(f"   Entities: {entities:,}")
    print(f"   Relationships: {relationships:,}")
    print(f"   Density: {density:.2f} relationships per entity")
    
    # 2. Sophistication Level Assessment
    if entities >= 100000:
        sophistication = "🚀 EXPERT-LEVEL AGI"
        description = "Approaching human expert capabilities"
    elif entities >= 50000:
        sophistication = "🧠 ADVANCED AGI"
        description = "Sophisticated reasoning and pattern recognition"
    elif entities >= 20000:
        sophistication = "⚡ INTELLIGENT AGI"
        description = "Complex multi-step reasoning active"
    elif entities >= 5000:
        sophistication = "🌟 EMERGING INTELLIGENCE"
        description = "Pattern recognition and basic reasoning"
    else:
        sophistication = "🌱 LEARNING FOUNDATION"
        description = "Building basic knowledge structures"
    
    print(f"\n🎯 Intelligence Level: {sophistication}")
    print(f"   {description}")
    
    # 3. Advanced Sophistication Metrics
    print(f"\n🔬 Advanced Intelligence Metrics:")
    
    try:
        # Multi-hop reasoning capability
        multi_hop_query = """
        MATCH path = (a:Entity)-[r1]->(b:Entity)-[r2]->(c:Entity)-[r3]->(d:Entity)
        WHERE r1.confidence > 0.7 AND r2.confidence > 0.7 AND r3.confidence > 0.7
        RETURN count(path) as three_hop_chains
        """
        result = kg.db.execute_query(multi_hop_query, {})
        three_hop_chains = result[0]['three_hop_chains'] if result else 0
        print(f"   🔗 3-Hop Reasoning Chains: {three_hop_chains:,}")
        
        # Cross-domain connections
        cross_domain_query = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.domain IS NOT NULL AND b.domain IS NOT NULL AND a.domain <> b.domain
        RETURN count(r) as cross_domain_connections
        """
        result = kg.db.execute_query(cross_domain_query, {})
        cross_domain = result[0]['cross_domain_connections'] if result else 0
        print(f"   🌐 Cross-Domain Connections: {cross_domain:,}")
        
        # High-confidence knowledge
        high_confidence_query = """
        MATCH ()-[r]->()
        WHERE r.confidence > 0.8
        RETURN count(r) as high_confidence_edges
        """
        result = kg.db.execute_query(high_confidence_query, {})
        high_confidence = result[0]['high_confidence_edges'] if result else 0
        confidence_ratio = (high_confidence / relationships * 100) if relationships > 0 else 0
        print(f"   ✅ High-Confidence Knowledge: {high_confidence:,} ({confidence_ratio:.1f}%)")
        
        # Complexity indicators
        complex_patterns_query = """
        MATCH (n:Entity)
        WITH n, size((n)-[]->()) as out_degree, size((n)<-[]-()) as in_degree
        WHERE out_degree >= 5 AND in_degree >= 5
        RETURN count(n) as hub_entities
        """
        result = kg.db.execute_query(complex_patterns_query, {})
        hub_entities = result[0]['hub_entities'] if result else 0
        print(f"   🎯 Knowledge Hubs (5+ connections): {hub_entities:,}")
        
    except Exception as e:
        print(f"   ⚠️ Advanced metrics unavailable: {e}")
    
    # 4. Intelligence Capability Assessment
    print(f"\n🚀 Current Capabilities Assessment:")
    
    if three_hop_chains > 1000:
        print("   ✅ COMPLEX REASONING: Multi-step causal inference active")
    elif three_hop_chains > 100:
        print("   ⚡ INTERMEDIATE REASONING: Basic causal chains established")
    else:
        print("   🌱 FOUNDATIONAL REASONING: Building reasoning pathways")
    
    if cross_domain > 500:
        print("   ✅ CROSS-DOMAIN TRANSFER: Knowledge synthesis across domains")
    elif cross_domain > 50:
        print("   ⚡ EMERGING TRANSFER: Some cross-domain connections")
    else:
        print("   🌱 DOMAIN-SPECIFIC: Learning within individual domains")
    
    if confidence_ratio > 70:
        print("   ✅ HIGH-QUALITY LEARNING: Confident knowledge formation")
    elif confidence_ratio > 50:
        print("   ⚡ RELIABLE LEARNING: Good knowledge confidence")
    else:
        print("   🌱 EXPLORATORY LEARNING: Building confidence in knowledge")
    
    if hub_entities > 1000:
        print("   ✅ HIERARCHICAL STRUCTURE: Complex knowledge organization")
    elif hub_entities > 100:
        print("   ⚡ EMERGING STRUCTURE: Knowledge hubs forming")
    else:
        print("   🌱 BASIC STRUCTURE: Linear knowledge connections")
    
    # 5. Practical Intelligence Estimate
    print(f"\n💡 Practical Intelligence Estimate:")
    
    if entities >= 80000 and three_hop_chains > 1000 and cross_domain > 500:
        print("   🎯 READY for complex problem solving")
        print("   🎯 CAPABLE of strategic planning assistance")
        print("   🎯 SUITABLE for domain expert consultation")
        
    elif entities >= 50000 and three_hop_chains > 500:
        print("   🎯 READY for pattern recognition tasks")
        print("   🎯 CAPABLE of multi-step reasoning")
        print("   🎯 SUITABLE for decision support systems")
        
    elif entities >= 20000:
        print("   🎯 READY for basic reasoning tasks")
        print("   🎯 CAPABLE of simple pattern detection")
        print("   🎯 SUITABLE for anomaly detection")
        
    else:
        print("   🌱 BUILDING foundational intelligence")
        print("   🌱 DEVELOPING basic reasoning capabilities")
        print("   🌱 PREPARING for practical applications")
    
    # Clean up connection
    try:
        if hasattr(kg, 'close'):
            kg.close()
        elif hasattr(kg, 'db') and hasattr(kg.db, 'close'):
            kg.db.close()
    except:
        pass
    print(f"\n✨ Analysis complete! Your AGI shows {sophistication.lower()}")

if __name__ == "__main__":
    try:
        analyze_intelligence_level()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
