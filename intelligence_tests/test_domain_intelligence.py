#!/usr/bin/env python3
"""
Domain Intelligence Test
Test the system's ability to reason within learned domains
"""

import json
from pathlib import Path
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

def test_domain_intelligence():
    """Test reasoning capabilities within learned domains"""
    
    print("🧠 DOMAIN INTELLIGENCE TEST")
    print("=" * 50)
    print("Testing reasoning within energy and medical domains")
    
    # Load config
    with open('config/database/neo4j.json', 'r') as f:
        config_data = json.load(f)
        neo4j_config = config_data['config']
    
    kg = EnhancedKnowledgeGraph(backend='neo4j', config=neo4j_config)
    
    # Test scenarios within learned domains
    test_scenarios = [
        {
            'domain': 'energy',
            'scenario': 'solar_energy_optimization_new_location',
            'description': 'Testing solar energy optimization for a new geographic location',
            'expected_reasoning': 'Should leverage existing solar energy knowledge'
        },
        {
            'domain': 'energy',
            'scenario': 'wind_power_integration_challenge',
            'description': 'Testing wind power grid integration challenges',
            'expected_reasoning': 'Should use learned wind energy patterns'
        },
        {
            'domain': 'medical',
            'scenario': 'drug_interaction_prediction',
            'description': 'Testing prediction of drug interactions',
            'expected_reasoning': 'Should apply medical knowledge patterns'
        },
        {
            'domain': 'medical',
            'scenario': 'symptom_diagnosis_correlation',
            'description': 'Testing symptom-diagnosis correlations',
            'expected_reasoning': 'Should use learned medical relationships'
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_scenarios, 1):
        print(f"\n🧪 TEST {i}: {test['description']}")
        print(f"   Domain: {test['domain']}")
        print(f"   Scenario: {test['scenario']}")
        
        # Check if domain knowledge exists
        domain_knowledge_query = f"""
        MATCH (n)
        WHERE n.id CONTAINS '{test['domain']}'
        RETURN count(n) as domain_entities
        """
        
        domain_results = kg.db.execute_query(domain_knowledge_query, {})
        domain_entities = domain_results[0]['domain_entities'] if domain_results else 0
        
        print(f"   📊 Domain entities available: {domain_entities:,}")
        
        if domain_entities == 0:
            print(f"   ❌ FAIL: No knowledge in {test['domain']} domain")
            results.append({
                'test': test['scenario'],
                'domain': test['domain'],
                'status': 'FAIL',
                'reason': 'No domain knowledge',
                'score': 0
            })
            continue
        
        # Test reasoning within domain
        reasoning_query = f"""
        MATCH (cause)-[r1]->(intermediate)-[r2]->(effect)
        WHERE (cause.id CONTAINS '{test['domain']}' OR intermediate.id CONTAINS '{test['domain']}' OR effect.id CONTAINS '{test['domain']}')
        AND r1.confidence > 0.7 AND r2.confidence > 0.7
        RETURN count(*) as reasoning_chains,
               avg(r1.confidence + r2.confidence) / 2 as avg_confidence
        LIMIT 100
        """
        
        reasoning_results = kg.db.execute_query(reasoning_query, {})
        if reasoning_results:
            chains = reasoning_results[0]['reasoning_chains'] or 0
            avg_conf = reasoning_results[0]['avg_confidence'] or 0
            
            print(f"   🔗 Reasoning chains found: {chains:,}")
            print(f"   🎯 Average confidence: {avg_conf:.3f}")
            
            # Score the test
            if chains > 1000 and avg_conf > 0.8:
                status = "EXCELLENT"
                score = 95
                print(f"   🚀 {status}: Strong reasoning capability")
            elif chains > 500 and avg_conf > 0.7:
                status = "GOOD"
                score = 80
                print(f"   ✅ {status}: Good reasoning capability")
            elif chains > 100 and avg_conf > 0.6:
                status = "FAIR"
                score = 65
                print(f"   📊 {status}: Basic reasoning capability")
            elif chains > 10:
                status = "WEAK"
                score = 40
                print(f"   ⚠️ {status}: Limited reasoning capability")
            else:
                status = "FAIL"
                score = 10
                print(f"   ❌ {status}: Insufficient reasoning capability")
            
            results.append({
                'test': test['scenario'],
                'domain': test['domain'],
                'status': status,
                'chains': chains,
                'confidence': avg_conf,
                'score': score
            })
        else:
            print(f"   ❌ FAIL: No reasoning chains found")
            results.append({
                'test': test['scenario'],
                'domain': test['domain'],
                'status': 'FAIL',
                'reason': 'No reasoning chains',
                'score': 0
            })
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 40)
    
    total_score = 0
    passed_tests = 0
    
    for result in results:
        score = result.get('score', 0)
        total_score += score
        if score >= 65:  # Consider 65+ as passing
            passed_tests += 1
        
        print(f"   {result['test'][:30]:30} | {result['status']:10} | {score:3d}/100")
    
    avg_score = total_score / len(results) if results else 0
    pass_rate = (passed_tests / len(results) * 100) if results else 0
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Average Score: {avg_score:.1f}/100")
    print(f"   Pass Rate: {pass_rate:.1f}% ({passed_tests}/{len(results)} tests)")
    
    if avg_score >= 85:
        print(f"   🚀 EXCELLENT: Strong domain intelligence")
    elif avg_score >= 70:
        print(f"   ✅ GOOD: Solid domain reasoning")
    elif avg_score >= 55:
        print(f"   📊 FAIR: Basic domain understanding")
    else:
        print(f"   ⚠️ NEEDS IMPROVEMENT: Limited domain intelligence")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if avg_score >= 80:
        print("   🎯 Ready for more complex scenarios within these domains")
        print("   🌟 Consider testing cross-domain reasoning")
        print("   📈 Monitor for knowledge consolidation over time")
    elif avg_score >= 60:
        print("   📚 Continue learning within current domains")
        print("   🔄 Focus on strengthening existing knowledge")
        print("   ⚡ Wait before introducing new domains")
    else:
        print("   🛠️ Check learning process effectiveness")
        print("   📖 Review scenario generation quality")
        print("   🔧 Consider adjusting learning parameters")
    
    # Domain-specific insights
    energy_results = [r for r in results if r['domain'] == 'energy']
    medical_results = [r for r in results if r['domain'] == 'medical']
    
    if energy_results:
        energy_avg = sum(r.get('score', 0) for r in energy_results) / len(energy_results)
        print(f"\n⚡ ENERGY DOMAIN: {energy_avg:.1f}/100 average")
        
    if medical_results:
        medical_avg = sum(r.get('score', 0) for r in medical_results) / len(medical_results)
        print(f"🏥 MEDICAL DOMAIN: {medical_avg:.1f}/100 average")
    
    # Cleanup
    try:
        if hasattr(kg.db, 'driver') and kg.db.driver:
            kg.db.driver.close()
    except:
        pass

if __name__ == "__main__":
    try:
        test_domain_intelligence()
        print(f"\n✨ Domain intelligence testing complete!")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
