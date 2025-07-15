#!/usr/bin/env python3
"""
Daily Progress Tracker for ICA Learning
Tracks key metrics over time and saves historical data
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from ica_framework.enhanced_knowledge_graph import EnhancedKnowledgeGraph

def daily_progress_tracker():
    """Track daily learning progress and save to CSV for trend analysis"""
    
    print("📊 DAILY PROGRESS TRACKER")
    print("=" * 50)
    
    # Load config
    with open('config/database/neo4j.json', 'r') as f:
        config_data = json.load(f)
        neo4j_config = config_data['config']
    
    kg = EnhancedKnowledgeGraph(backend='neo4j', config=neo4j_config)
    
    # Current timestamp
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H:%M:%S")
    
    print(f"📅 Tracking progress for: {date_str} {time_str}")
    
    # Get current statistics
    stats_query = """
    MATCH (n) 
    OPTIONAL MATCH (n)-[r]-()
    RETURN 
        count(DISTINCT n) as total_nodes,
        count(DISTINCT r) as total_relationships,
        count(DISTINCT labels(n)) as unique_node_types,
        count(DISTINCT type(r)) as unique_relationship_types
    """
    
    stats = kg.db.execute_query(stats_query, {})
    current_nodes = stats[0]['total_nodes']
    current_relationships = stats[0]['total_relationships']
    node_types = stats[0]['unique_node_types']
    rel_types = stats[0]['unique_relationship_types']
    
    # Calculate density
    density = current_relationships / current_nodes if current_nodes > 0 else 0
    
    # Get confidence metrics
    confidence_query = """
    MATCH ()-[r]->()
    RETURN 
        avg(r.confidence) as avg_confidence,
        count(CASE WHEN r.confidence > 0.8 THEN 1 END) as high_conf,
        count(CASE WHEN r.confidence > 0.9 THEN 1 END) as very_high_conf,
        count(r) as total_rels
    """
    
    conf_results = kg.db.execute_query(confidence_query, {})
    avg_conf = conf_results[0]['avg_confidence'] or 0
    high_conf = conf_results[0]['high_conf'] or 0
    very_high_conf = conf_results[0]['very_high_conf'] or 0
    
    # Get domain coverage
    domain_coverage_query = """
    MATCH (n)
    RETURN 
        count(CASE WHEN n.id CONTAINS 'energy' THEN 1 END) as energy_entities,
        count(CASE WHEN n.id CONTAINS 'medical' THEN 1 END) as medical_entities,
        count(CASE WHEN n.id CONTAINS 'environmental' THEN 1 END) as env_entities,
        count(CASE WHEN n.id CONTAINS 'manufacturing' THEN 1 END) as mfg_entities,
        count(CASE WHEN n.id CONTAINS 'transportation' THEN 1 END) as transport_entities,
        count(CASE WHEN n.id CONTAINS 'financial' THEN 1 END) as financial_entities
    """
    
    domain_results = kg.db.execute_query(domain_coverage_query, {})
    domains = domain_results[0] if domain_results else {}
    
    # Get complex reasoning chains
    complex_reasoning_query = """
    MATCH path = (a)-[r1]->(b)-[r2]->(c)-[r3]->(d)
    WHERE r1.confidence > 0.7 AND r2.confidence > 0.7 AND r3.confidence > 0.7
    RETURN count(*) as four_step_chains
    """
    
    complex_results = kg.db.execute_query(complex_reasoning_query, {})
    complex_chains = complex_results[0]['four_step_chains'] or 0
    
    # Prepare data row
    data_row = {
        'date': date_str,
        'time': time_str,
        'timestamp': timestamp.timestamp(),
        'total_nodes': current_nodes,
        'total_relationships': current_relationships,
        'node_types': node_types,
        'rel_types': rel_types,
        'density': density,
        'avg_confidence': avg_conf,
        'high_confidence_count': high_conf,
        'very_high_confidence_count': very_high_conf,
        'complex_chains': complex_chains,
        'energy_entities': domains.get('energy_entities', 0),
        'medical_entities': domains.get('medical_entities', 0),
        'env_entities': domains.get('env_entities', 0),
        'mfg_entities': domains.get('mfg_entities', 0),
        'transport_entities': domains.get('transport_entities', 0),
        'financial_entities': domains.get('financial_entities', 0)
    }
    
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Save to CSV
    csv_file = logs_dir / 'daily_progress.csv'
    file_exists = csv_file.exists()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_row.keys())
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data_row)
    
    print(f"✅ Progress data saved to: {csv_file}")
    
    # Display current metrics
    print(f"\n📈 CURRENT METRICS:")
    print(f"   Nodes: {current_nodes:,}")
    print(f"   Relationships: {current_relationships:,}")
    print(f"   Density: {density:.2f}")
    print(f"   Avg Confidence: {avg_conf:.3f}")
    print(f"   High Confidence: {high_conf:,} ({(high_conf/current_relationships*100):.1f}%)")
    print(f"   Complex Chains: {complex_chains:,}")
    
    # Show growth if we have previous data
    if file_exists:
        # Read previous entries to calculate growth
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if len(rows) >= 2:  # We have at least one previous entry
                prev_row = rows[-2]  # Second to last (last is current)
                
                prev_nodes = int(prev_row['total_nodes'])
                prev_relationships = int(prev_row['total_relationships'])
                prev_chains = int(prev_row['complex_chains'])
                
                node_growth = ((current_nodes - prev_nodes) / prev_nodes * 100) if prev_nodes > 0 else 0
                rel_growth = ((current_relationships - prev_relationships) / prev_relationships * 100) if prev_relationships > 0 else 0
                chain_growth = ((complex_chains - prev_chains) / prev_chains * 100) if prev_chains > 0 else 0
                
                print(f"\n📊 GROWTH SINCE LAST TRACKING:")
                print(f"   Nodes: {node_growth:+.1f}% (+{current_nodes - prev_nodes:,})")
                print(f"   Relationships: {rel_growth:+.1f}% (+{current_relationships - prev_relationships:,})")
                print(f"   Complex Chains: {chain_growth:+.1f}% (+{complex_chains - prev_chains:,})")
                
                # Growth rate assessment
                if node_growth > 10:
                    print("   🚀 HIGH GROWTH RATE")
                elif node_growth > 5:
                    print("   ✅ GOOD GROWTH RATE")
                elif node_growth > 1:
                    print("   📈 MODERATE GROWTH")
                elif node_growth > 0:
                    print("   🐌 SLOW GROWTH")
                else:
                    print("   ⚠️ NO GROWTH")
    
    # Cleanup
    try:
        if hasattr(kg.db, 'driver') and kg.db.driver:
            kg.db.driver.close()
    except:
        pass
    
    print(f"\n💡 TIP: Run this daily to track learning trends over time!")

def analyze_progress_trends():
    """Analyze trends from the daily progress CSV"""
    
    csv_file = Path('logs/daily_progress.csv')
    if not csv_file.exists():
        print("❌ No progress data found. Run daily tracker first.")
        return
    
    print("📈 PROGRESS TREND ANALYSIS")
    print("=" * 50)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if len(rows) < 2:
        print("❌ Need at least 2 data points for trend analysis")
        return
    
    # Get first and last entries
    first = rows[0]
    last = rows[-1]
    
    # Calculate overall growth
    days_elapsed = len(rows) - 1
    first_nodes = int(first['total_nodes'])
    last_nodes = int(last['total_nodes'])
    first_rels = int(first['total_relationships'])
    last_rels = int(last['total_relationships'])
    
    total_node_growth = ((last_nodes - first_nodes) / first_nodes * 100) if first_nodes > 0 else 0
    total_rel_growth = ((last_rels - first_rels) / first_rels * 100) if first_rels > 0 else 0
    
    avg_daily_node_growth = total_node_growth / days_elapsed if days_elapsed > 0 else 0
    avg_daily_rel_growth = total_rel_growth / days_elapsed if days_elapsed > 0 else 0
    
    print(f"📊 OVERALL TRENDS ({days_elapsed} tracking periods):")
    print(f"   Total Node Growth: {total_node_growth:.1f}%")
    print(f"   Total Relationship Growth: {total_rel_growth:.1f}%")
    print(f"   Average Daily Node Growth: {avg_daily_node_growth:.1f}%")
    print(f"   Average Daily Rel Growth: {avg_daily_rel_growth:.1f}%")
    
    # Show recent entries
    print(f"\n📅 RECENT PROGRESS:")
    for i, row in enumerate(rows[-5:]):  # Last 5 entries
        print(f"   {row['date']} {row['time']}: {int(row['total_nodes']):,} nodes, {int(row['total_relationships']):,} rels")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'trends':
        analyze_progress_trends()
    else:
        try:
            daily_progress_tracker()
        except Exception as e:
            print(f"❌ Tracking failed: {e}")
            import traceback
            traceback.print_exc()
