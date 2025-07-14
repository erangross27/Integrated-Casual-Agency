#!/usr/bin/env python3
"""
Monitor the continuous learning progress
Check if it's running and show current knowledge stats
"""

import json
import time
import subprocess
from pathlib import Path
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

def check_learning_status():
    """Check if continuous learning is running and show progress"""
    
    print("🔍 CONTINUOUS LEARNING STATUS CHECK")
    print("=" * 50)
    
    # Check if Python processes are running
    try:
        result = subprocess.run(
            ["powershell", "-Command", "Get-Process python | Measure-Object | Select-Object Count"],
            capture_output=True, text=True, check=True
        )
        
        # Extract count from PowerShell output
        output_lines = result.stdout.strip().split('\n')
        count_line = [line for line in output_lines if line.strip().isdigit()]
        if count_line:
            python_processes = int(count_line[0].strip())
            
            if python_processes >= 10:
                print(f"✅ LEARNING ACTIVE: {python_processes} Python processes running")
                print("   (Expected: 10 workers + 1 main process = 11 total)")
            elif python_processes > 0:
                print(f"⚠️ PARTIAL LEARNING: {python_processes} Python processes running")
                print("   (May be starting up or shutting down)")
            else:
                print("❌ LEARNING STOPPED: No Python processes found")
                return
        else:
            print("❓ UNCLEAR STATUS: Could not determine process count")
            
    except Exception as e:
        print(f"⚠️ Could not check processes: {e}")
    
    # Check current knowledge stats
    print(f"\n📊 CURRENT KNOWLEDGE BASE:")
    try:
        # Load config
        with open('config/database/neo4j.json', 'r') as f:
            config_data = json.load(f)
            neo4j_config = config_data['config']
        
        kg = EnhancedKnowledgeGraph(backend='neo4j', config=neo4j_config)
        
        # Quick stats query
        stats_query = """
        MATCH (n) 
        OPTIONAL MATCH (n)-[r]-()
        RETURN 
            count(DISTINCT n) as total_nodes,
            count(DISTINCT r) as total_relationships
        """
        
        stats = kg.db.execute_query(stats_query, {})
        if stats:
            nodes = stats[0]['total_nodes']
            rels = stats[0]['total_relationships']
            print(f"   Nodes: {nodes:,}")
            print(f"   Relationships: {rels:,}")
            
            # Calculate learning rate (rough estimate)
            if nodes > 139000:  # Starting point
                new_nodes = nodes - 139000
                print(f"   🆕 New nodes since start: {new_nodes:,}")
            
        # Cleanup
        try:
            if hasattr(kg.db, 'driver') and kg.db.driver:
                kg.db.driver.close()
        except:
            pass
            
    except Exception as e:
        print(f"⚠️ Could not check database: {e}")
    
    print(f"\n🎯 INTELLIGENCE TESTING:")
    print("   To check learning progress, run:")
    print("   python intelligence_tests/test_novel_intelligence.py")
    print("   (Currently: 0/500, should improve over months)")
    
    print(f"\n🛑 TO STOP LEARNING:")
    print("   Get-Process python | Stop-Process -Force")
    print("   (This will stop all Python processes)")

if __name__ == "__main__":
    check_learning_status()
