#!/usr/bin/env python3
"""
Real-World AGI Intelligence Test
Tests actual problem-solving capabilities with real scenarios
"""

import sys
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

def test_real_world_intelligence():
    """Test AGI with actual real-world problems"""
    
    print("🧠 REAL-WORLD AGI Intelligence Test")
    print("=" * 50)
    
    # Load Neo4j config (handle both direct run and from intelligence_tests folder)
    config_path = Path('../config/database/neo4j.json')
    if not config_path.exists():
        config_path = Path('config/database/neo4j.json')
    
    if not config_path.exists():
        print("❌ Neo4j config not found. Please run 'python setup.py database' first.")
        return 0
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
        neo4j_config = config_data['config']
    
    # Connect to knowledge graph
    kg = EnhancedKnowledgeGraph(backend='neo4j', config=neo4j_config)
    
    # Test 1: Can it find energy efficiency patterns?
    print("\n🔋 TEST 1: Energy Efficiency Problem")
    print("Question: What reduces energy consumption?")
    
    energy_query = """
    MATCH (cause:Entity)-[r]->(effect:Entity)
    WHERE (cause.id CONTAINS 'motion' OR cause.id CONTAINS 'sensor' OR cause.id CONTAINS 'light')
    AND (effect.id CONTAINS 'energy' OR effect.id CONTAINS 'power' OR effect.id CONTAINS 'consumption')
    AND r.confidence > 0.7
    RETURN cause.id as cause, r.type as relationship, effect.id as effect, r.confidence as confidence
    ORDER BY r.confidence DESC
    LIMIT 10
    """
    
    try:
        results = kg.db.execute_query(energy_query, {})
        if results:
            print("✅ Found energy efficiency patterns:")
            for result in results[:5]:
                print(f"   {result['cause']} → {result['relationship']} → {result['effect']} (confidence: {result['confidence']:.2f})")
            score_1 = min(len(results), 5) * 20  # Max 100 points
        else:
            print("❌ No energy efficiency patterns found")
            score_1 = 0
    except Exception as e:
        print(f"❌ Query failed: {e}")
        score_1 = 0
    
    # Test 2: Can it understand cause-effect chains?
    print(f"\n🔗 TEST 2: Causal Chain Reasoning")
    print("Question: Find 3-step cause-effect chains")
    
    causal_query = """
    MATCH path = (a:Entity)-[r1]->(b:Entity)-[r2]->(c:Entity)
    WHERE r1.confidence > 0.8 AND r2.confidence > 0.8
    AND a.id <> c.id
    RETURN a.id as start, r1.type as step1, b.id as middle, r2.type as step2, c.id as end
    ORDER BY r1.confidence + r2.confidence DESC
    LIMIT 10
    """
    
    try:
        results = kg.db.execute_query(causal_query, {})
        if results:
            print("✅ Found causal reasoning chains:")
            for result in results[:3]:
                print(f"   {result['start']} → {result['step1']} → {result['middle']} → {result['step2']} → {result['end']}")
            score_2 = min(len(results), 5) * 20  # Max 100 points
        else:
            print("❌ No causal chains found")
            score_2 = 0
    except Exception as e:
        print(f"❌ Query failed: {e}")
        score_2 = 0
    
    # Test 3: Can it find optimization opportunities?
    print(f"\n⚡ TEST 3: System Optimization")
    print("Question: What can be optimized or improved?")
    
    optimization_query = """
    MATCH (entity:Entity)-[r]->(target:Entity)
    WHERE (r.type CONTAINS 'optimize' OR r.type CONTAINS 'improve' OR r.type CONTAINS 'enhance')
    AND r.confidence > 0.6
    RETURN entity.id as system, r.type as optimization, target.id as target, r.confidence as confidence
    ORDER BY r.confidence DESC
    LIMIT 10
    """
    
    try:
        results = kg.db.execute_query(optimization_query, {})
        if results:
            print("✅ Found optimization opportunities:")
            for result in results[:3]:
                print(f"   {result['system']} can {result['optimization']} {result['target']} (confidence: {result['confidence']:.2f})")
            score_3 = min(len(results), 5) * 20  # Max 100 points
        else:
            print("❌ No optimization patterns found")
            score_3 = 0
    except Exception as e:
        print(f"❌ Query failed: {e}")
        score_3 = 0
    
    # Test 4: Can it identify safety or security concerns?
    print(f"\n🛡️ TEST 4: Safety & Security Analysis")
    print("Question: What causes failures or problems?")
    
    safety_query = """
    MATCH (cause:Entity)-[r]->(problem:Entity)
    WHERE (problem.id CONTAINS 'failure' OR problem.id CONTAINS 'error' OR problem.id CONTAINS 'fault' OR problem.id CONTAINS 'problem')
    AND r.confidence > 0.6
    RETURN cause.id as cause, r.type as relationship, problem.id as problem, r.confidence as confidence
    ORDER BY r.confidence DESC
    LIMIT 10
    """
    
    try:
        results = kg.db.execute_query(safety_query, {})
        if results:
            print("✅ Found safety/security patterns:")
            for result in results[:3]:
                print(f"   {result['cause']} → {result['relationship']} → {result['problem']} (confidence: {result['confidence']:.2f})")
            score_4 = min(len(results), 5) * 20  # Max 100 points
        else:
            print("❌ No safety patterns found")
            score_4 = 0
    except Exception as e:
        print(f"❌ Query failed: {e}")
        score_4 = 0
    
    # Test 5: Can it make predictions based on patterns?
    print(f"\n🔮 TEST 5: Predictive Intelligence")
    print("Question: What leads to specific outcomes?")
    
    prediction_query = """
    MATCH (input:Entity)-[r]->(output:Entity)
    WHERE r.confidence > 0.8
    AND (output.id CONTAINS 'increase' OR output.id CONTAINS 'decrease' OR output.id CONTAINS 'change')
    RETURN input.id as input, r.type as predicts, output.id as outcome, r.confidence as confidence
    ORDER BY r.confidence DESC
    LIMIT 10
    """
    
    try:
        results = kg.db.execute_query(prediction_query, {})
        if results:
            print("✅ Found predictive patterns:")
            for result in results[:3]:
                print(f"   {result['input']} → {result['predicts']} → {result['outcome']} (confidence: {result['confidence']:.2f})")
            score_5 = min(len(results), 5) * 20  # Max 100 points
        else:
            print("❌ No predictive patterns found")
            score_5 = 0
    except Exception as e:
        print(f"❌ Query failed: {e}")
        score_5 = 0
    
    # Calculate overall intelligence score
    total_score = score_1 + score_2 + score_3 + score_4 + score_5
    
    print(f"\n📊 REAL-WORLD INTELLIGENCE ASSESSMENT")
    print("=" * 50)
    print(f"🔋 Energy Efficiency: {score_1}/100")
    print(f"🔗 Causal Reasoning: {score_2}/100") 
    print(f"⚡ Optimization: {score_3}/100")
    print(f"🛡️ Safety Analysis: {score_4}/100")
    print(f"🔮 Predictive Intelligence: {score_5}/100")
    print("-" * 30)
    print(f"🎯 TOTAL SCORE: {total_score}/500")
    
    # Intelligence classification based on real performance
    if total_score >= 400:
        level = "🚀 EXPERT-LEVEL REAL AGI"
        description = "Demonstrates human-expert problem-solving"
    elif total_score >= 300:
        level = "🧠 ADVANCED REAL INTELLIGENCE"
        description = "Strong real-world problem-solving capabilities"
    elif total_score >= 200:
        level = "⚡ PRACTICAL INTELLIGENCE"
        description = "Useful for real-world applications"
    elif total_score >= 100:
        level = "🌟 EMERGING INTELLIGENCE"
        description = "Basic real-world pattern recognition"
    else:
        level = "🌱 LEARNING FOUNDATION"
        description = "Still building practical knowledge"
    
    print(f"\n🎯 REAL INTELLIGENCE LEVEL: {level}")
    print(f"   {description}")
    
    # Specific recommendations
    print(f"\n💡 INTELLIGENCE INSIGHTS:")
    
    if score_1 < 50:
        print("   🔋 Needs more energy efficiency learning")
    if score_2 < 50:
        print("   🔗 Causal reasoning needs development")
    if score_3 < 50:
        print("   ⚡ Optimization patterns are weak")
    if score_4 < 50:
        print("   🛡️ Safety analysis capabilities limited")
    if score_5 < 50:
        print("   🔮 Predictive intelligence underdeveloped")
    
    if total_score >= 200:
        print("   ✅ System shows genuine practical intelligence!")
        print("   ✅ Ready for real-world problem-solving tasks")
    elif total_score >= 100:
        print("   ⚡ System shows promising intelligence foundations")
        print("   ⚡ Continue learning for practical applications")
    else:
        print("   🌱 System needs more diverse real-world training")
        print("   🌱 Current knowledge is primarily simulated")
    
    # Cleanup
    try:
        if hasattr(kg, 'close'):
            kg.close()
        elif hasattr(kg, 'db') and hasattr(kg.db, 'close'):
            kg.db.close()
    except:
        pass
    
    return total_score

if __name__ == "__main__":
    try:
        score = test_real_world_intelligence()
        print(f"\n✨ Real-world intelligence test complete!")
        print(f"   Your AGI scored {score}/500 on practical problems")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
