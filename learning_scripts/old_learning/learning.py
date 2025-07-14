#!/usr/bin/env python3
"""
Simplified ICA Framework Learning Script
Now uses the refactored learning module components
"""

import argparse
import json
import multiprocessing as mp
import psutil
from pathlib import Path
import sys

# Add the parent directory to the path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Suppress all logging
import logging
import warnings
import os

logging.disable(logging.CRITICAL)
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

from ica_framework.learning import ContinuousLearning


def main():
    """Main function to run continuous learning with database options"""
    parser = argparse.ArgumentParser(description='ICA Framework Continuous Learning')
    parser.add_argument('--backend', choices=['memory', 'neo4j'], default='neo4j',
                       help='Database backend to use (default: neo4j)')
    parser.add_argument('--neo4j-uri', default=None,
                       help='Neo4j URI (overrides config file)')
    parser.add_argument('--neo4j-user', default=None,
                       help='Neo4j username (overrides config file)')
    parser.add_argument('--neo4j-password', default=None,
                       help='Neo4j password (overrides config file)')
    parser.add_argument('--neo4j-database', default=None,
                       help='Neo4j database name (overrides config file)')
    
    # Parallel processing arguments
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel processing (default: True)')
    parser.add_argument('--no-parallel', action='store_true', default=False,
                       help='Disable parallel processing (use sequential mode)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: 75%% of CPU cores)')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Number of scenarios per batch (default: 20)')
    parser.add_argument('--continuous', action='store_true', default=False,
                       help='Enable continuous parallel mode (no batching, constant worker pool)')
    
    args = parser.parse_args()
    
    # Handle parallel processing flags
    enable_parallel = args.parallel and not args.no_parallel
    continuous_mode = args.continuous and enable_parallel  # Continuous mode requires parallel
    
    # Configure database
    database_config = {}
    
    if args.backend == 'neo4j':
        # Try to load from config file first
        config_file = Path("config/database/neo4j.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)['config']
                
                database_config = {
                    'uri': args.neo4j_uri or file_config.get('uri', 'neo4j://127.0.0.1:7687'),
                    'username': args.neo4j_user or file_config.get('username', 'neo4j'),
                    'password': args.neo4j_password or file_config.get('password', 'password'),
                    'database': args.neo4j_database or file_config.get('database', 'neo4j')
                }
                
                print(f"🗄️ Neo4j Configuration (from {config_file}):")
            except Exception as e:
                print(f"⚠️ Failed to load config file: {e}")
                print("Using command line arguments...")
                database_config = {
                    'uri': args.neo4j_uri or 'neo4j://127.0.0.1:7687',
                    'username': args.neo4j_user or 'neo4j',
                    'password': args.neo4j_password or 'password',
                    'database': args.neo4j_database or 'neo4j'
                }
        else:
            print(f"⚠️ Config file not found: {config_file}")
            print("Using command line arguments...")
            database_config = {
                'uri': args.neo4j_uri or 'neo4j://127.0.0.1:7687',
                'username': args.neo4j_user or 'neo4j',
                'password': args.neo4j_password or 'password',
                'database': args.neo4j_database or 'neo4j'
            }
        
        print(f"   URI: {database_config['uri']}")
        print(f"   Database: {database_config['database']}")
        print(f"   User: {database_config['username']}")
        print()
    
    # Show parallel processing configuration
    if enable_parallel:
        workers = args.workers or max(1, int(mp.cpu_count() * (0.9 if continuous_mode else 0.75)))
        print(f"⚡ Parallel Processing Configuration:")
        if continuous_mode:
            print(f"   Mode: CONTINUOUS PARALLEL (no batching)")
            print(f"   Workers: {workers} (of {mp.cpu_count()} CPU cores) - processing individually")
            print(f"   Expected utilization: Maximum constant processing across all cores")
        else:
            print(f"   Mode: BATCH PARALLEL")
            print(f"   Workers: {workers} (of {mp.cpu_count()} CPU cores)")
            print(f"   Batch size: {args.batch_size} scenarios per batch")
            print(f"   Expected speedup: ~{workers}x faster processing")
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            print(f"   System memory: {memory_gb:.1f}GB")
        except:
            pass
        print()
    else:
        print("🔄 Sequential Processing Mode")
        print()
    
    # Create learning system with specified backend
    learner = ContinuousLearning(
        database_backend=args.backend,
        database_config=database_config,
        enable_parallel=enable_parallel,
        num_workers=args.workers,
        batch_size=args.batch_size,
        continuous_mode=continuous_mode
    )
    
    learner.run_continuous_learning()


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()
