#!/usr/bin/env python3
"""
AGI Progress Monitor
Automatically tests AGI intelligence at regular intervals
Shows progress trends and alerts for significant improvements
"""

import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

PROJECT_ROOT = Path(__file__).parent.parent

def run_intelligence_test():
    """Run the AGI intelligence test"""
    print(f"\n🧠 Running AGI Intelligence Test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Run the intelligence tester
        result = subprocess.run([
            sys.executable, 
            str(PROJECT_ROOT / "scripts" / "real_agi_intelligence_tester.py")
        ], capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Intelligence test completed successfully!")
        else:
            print("⚠️ Intelligence test had issues")
            
    except Exception as e:
        print(f"❌ Error running intelligence test: {e}")

def check_should_test():
    """Check if enough time has passed or progress made to warrant testing"""
    history_file = PROJECT_ROOT / "agi_checkpoints" / "intelligence_history.json"
    
    if not history_file.exists():
        return True, "No previous tests found"
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        if not history.get('test_history'):
            return True, "No test history"
        
        last_test = history['test_history'][-1]
        last_test_time = datetime.fromisoformat(last_test['timestamp'])
        time_since = datetime.now() - last_test_time
        
        # Test every 2 hours minimum
        if time_since > timedelta(hours=2):
            return True, f"Last test was {time_since} ago"
        
        return False, f"Last test was only {time_since} ago (waiting for 2+ hours)"
        
    except Exception as e:
        print(f"⚠️ Error checking test history: {e}")
        return True, "Error reading history"

def monitor_agi_progress(test_interval_hours=2, max_runtime_hours=24):
    """Monitor AGI progress with automatic testing"""
    print(f"🎯 AGI PROGRESS MONITOR STARTED")
    print(f"=" * 40)
    print(f"⏰ Test Interval: Every {test_interval_hours} hours")
    print(f"⏱️ Max Runtime: {max_runtime_hours} hours")
    print(f"🌐 W&B Dashboard: https://wandb.ai/erangross/true-agi-system/weave")
    print()
    
    start_time = datetime.now()
    test_count = 0
    
    while True:
        # Check if we should stop
        runtime = datetime.now() - start_time
        if runtime > timedelta(hours=max_runtime_hours):
            print(f"⏰ Stopping monitor after {runtime}")
            break
        
        # Check if we should test
        should_test, reason = check_should_test()
        
        current_time = datetime.now().strftime('%H:%M:%S')
        print(f"[{current_time}] Monitor Check: {reason}")
        
        if should_test:
            test_count += 1
            print(f"\n🔍 RUNNING TEST #{test_count}")
            run_intelligence_test()
            print(f"\n📊 Check W&B dashboard for latest metrics!")
            print(f"🔗 https://wandb.ai/erangross/true-agi-system/weave")
        
        # Wait before next check (check every 30 minutes)
        print(f"⏳ Waiting 30 minutes before next check...")
        time.sleep(30 * 60)  # 30 minutes

def main():
    """Main function with options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor AGI Progress')
    parser.add_argument('--test-interval', type=int, default=2, 
                       help='Hours between tests (default: 2)')
    parser.add_argument('--max-runtime', type=int, default=24,
                       help='Maximum hours to run monitor (default: 24)')
    parser.add_argument('--test-now', action='store_true',
                       help='Run a test immediately and exit')
    
    args = parser.parse_args()
    
    if args.test_now:
        print("🧠 Running immediate AGI intelligence test...")
        run_intelligence_test()
        return
    
    try:
        monitor_agi_progress(args.test_interval, args.max_runtime)
    except KeyboardInterrupt:
        print(f"\n⏹️ Monitor stopped by user")
        print(f"📊 Check your progress in W&B dashboard!")

if __name__ == "__main__":
    main()
