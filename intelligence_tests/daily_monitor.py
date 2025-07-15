#!/usr/bin/env python3
"""
Daily Learning Monitor
Quick daily check of AGI learning progress
"""

import json
import datetime
from pathlib import Path
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

def daily_progress_check():
    """Quick daily learning progress check"""
    
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"📅 DAILY LEARNING CHECK - {today}")
    print("=" * 50)
    
    # Load config
    with open('config/database/neo4j.json', 'r') as f:
        config_data = json.load(f)
        neo4j_config = config_data['config']
    
    kg = EnhancedKnowledgeGraph(backend='neo4j', config=neo4j_config)
    
    # Quick stats
    stats_query = """
    MATCH (n) 
    OPTIONAL MATCH (n)-[r]-()
    RETURN 
        count(DISTINCT n) as nodes,
        count(DISTINCT r) as relationships
    """
    
    stats = kg.db.execute_query(stats_query, {})
    nodes = stats[0]['nodes']
    relationships = stats[0]['relationships']
    
    # Baseline from July 15, 2025 morning
    baseline_nodes = 139000
    baseline_rels = 610000
    
    print(f"📊 KNOWLEDGE BASE:")
    print(f"   Nodes: {nodes:,} (+{nodes - baseline_nodes:,})")
    print(f"   Relationships: {relationships:,} (+{relationships - baseline_rels:,})")
    print(f"   Growth Rate: {((nodes/baseline_nodes - 1)*100):.1f}%")
    
    # Quick intelligence check
    progress_query = """
    MATCH path = (a)-[r1]->(b)-[r2]->(c)-[r3]->(d)
    WHERE r1.confidence > 0.7 AND r2.confidence > 0.7 AND r3.confidence > 0.7
    RETURN count(*) as complex_chains
    """
    
    progress = kg.db.execute_query(progress_query, {})
    chains = progress[0]['complex_chains'] if progress else 0
    
    confidence_query = """
    MATCH ()-[r]->()
    WHERE r.confidence > 0.9
    RETURN count(r) as high_conf
    """
    
    conf_results = kg.db.execute_query(confidence_query, {})
    high_conf = conf_results[0]['high_conf'] if conf_results else 0
    
    print(f"\n🧠 INTELLIGENCE METRICS:")
    print(f"   Complex Reasoning Chains: {chains:,}")
    print(f"   High-Confidence Relations: {high_conf:,}")
    
    # Calculate daily progress score (simplified)
    progress_score = min(chains // 1000, 100) + min(high_conf // 1000, 100)
    
    print(f"\n📈 DAILY PROGRESS SCORE: {progress_score}/200")
    
    if progress_score > 150:
        print("   🚀 EXCELLENT: Strong learning progress")
    elif progress_score > 100:
        print("   ✅ GOOD: Steady learning improvement")
    elif progress_score > 50:
        print("   📈 FAIR: Learning in progress")
    else:
        print("   ⚠️ SLOW: May need attention")
    
    # Learning status
    import subprocess
    try:
        result = subprocess.run(
            ["powershell", "-Command", "Get-Process python | Measure-Object | Select-Object Count"],
            capture_output=True, text=True, check=True
        )
        output_lines = result.stdout.strip().split('\n')
        count_line = [line for line in output_lines if line.strip().isdigit()]
        if count_line:
            python_processes = int(count_line[0].strip())
            print(f"\n🔄 LEARNING STATUS:")
            if python_processes >= 9:
                print(f"   ✅ ACTIVE: {python_processes} workers running")
            else:
                print(f"   ⚠️ INACTIVE: Only {python_processes} processes")
        else:
            print(f"\n🔄 LEARNING STATUS: Unknown")
    except:
        print(f"\n🔄 LEARNING STATUS: Could not check")
    
    print(f"\n💡 QUICK INSIGHTS:")
    print(f"   • Knowledge grew by {((nodes/baseline_nodes - 1)*100):.1f}% since baseline")
    print(f"   • {chains:,} complex reasoning patterns built")
    print(f"   • {high_conf:,} high-confidence relationships established")
    
    # Cleanup
    try:
        if hasattr(kg.db, 'driver') and kg.db.driver:
            kg.db.driver.close()
    except:
        pass

if __name__ == "__main__":
    try:
        daily_progress_check()
    except Exception as e:
        print(f"❌ Daily check failed: {e}")
        import traceback
        traceback.print_exc()
