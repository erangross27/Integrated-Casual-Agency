#!/usr/bin/env python3
"""
AGI Progress Summary
Shows the latest test results and progress in a clean format
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

def show_latest_progress():
    """Show the latest AGI test results and progress"""
    
    # Check for intelligence history
    history_file = PROJECT_ROOT / "agi_checkpoints" / "intelligence_history.json"
    
    if not history_file.exists():
        print("❌ No intelligence test history found.")
        print("💡 Run a test first: python scripts/real_agi_intelligence_tester.py")
        return
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        test_history = history.get('test_history', [])
        
        if not test_history:
            print("❌ No test results found in history.")
            return
        
        # Get the latest test
        latest_test = test_history[-1]
        
        print(f"🧠 LATEST AGI INTELLIGENCE TEST RESULTS")
        print(f"=" * 45)
        print(f"📅 Test Date: {latest_test['timestamp']}")
        print(f"🔬 Session: {latest_test['session']}")
        print()
        
        # Show overall intelligence
        overall_intel = latest_test.get('overall_intelligence', 0)
        results = latest_test.get('results', {})
        
        print(f"🎯 OVERALL INTELLIGENCE: {overall_intel:.1f}/10")
        
        if overall_intel >= 8.0:
            level = "🤖 ADVANCED AGI - Excellent physics reasoning"
        elif overall_intel >= 6.0:
            level = "🚀 COMPETENT AGI - Strong pattern recognition"
        elif overall_intel >= 4.0:
            level = "📚 LEARNING AGI - Developing understanding"
        else:
            level = "🌱 EARLY AGI - Basic pattern detection"
        
        print(f"📊 AGI Level: {level}")
        print()
        
        # Show individual test results if available
        if isinstance(results, list):
            individual_tests = results
        else:
            individual_tests = results.get('individual_tests', [])
        
        if individual_tests:
            print(f"🧪 INDIVIDUAL TEST RESULTS:")
            print(f"-" * 30)
            for i, test in enumerate(individual_tests, 1):
                name = test.get('test', f'Test {i}')
                understanding = test.get('understanding_level', 0)
                interpretation = test.get('interpretation', 'No interpretation')
                
                print(f"{i:2d}. {name}")
                print(f"    Understanding: {understanding:.1f}/10")
                print(f"    Analysis: {interpretation}")
                print()
        
        # Show training progress
        training_info = latest_test.get('training_info', {})
        if training_info:
            print(f"📚 TRAINING PROGRESS:")
            print(f"-" * 20)
            concepts = training_info.get('concepts', 0)
            hypotheses = training_info.get('hypotheses', 0)
            epochs = training_info.get('training_epochs', 0)
            
            print(f"💡 Concepts Learned: {concepts:,}")
            print(f"🧪 Hypotheses Formed: {hypotheses:,}")
            print(f"🔄 Training Epochs: {epochs}")
            print()
        
        # Show progress comparison if we have multiple tests
        if len(test_history) > 1:
            prev_test = test_history[-2]
            prev_intel = prev_test.get('overall_intelligence', 0)
            intel_change = overall_intel - prev_intel
            
            print(f"📈 PROGRESS SINCE LAST TEST:")
            print(f"-" * 30)
            print(f"Previous Intelligence: {prev_intel:.2f}/10")
            print(f"Current Intelligence:  {overall_intel:.2f}/10")
            
            if intel_change > 0:
                print(f"📊 Improvement: +{intel_change:.2f} ({intel_change/prev_intel*100:.1f}%)")
                print(f"🚀 Intelligence is improving! Keep training!")
            elif intel_change < 0:
                print(f"📉 Decrease: {intel_change:.2f} ({intel_change/prev_intel*100:.1f}%)")
                print(f"⚠️ Temporary decrease - this can happen during learning")
            else:
                print(f"➡️ No change in overall intelligence")
                print(f"🔄 May need more training time for next breakthrough")
            
            # Training progress comparison
            prev_training = prev_test.get('training_info', {})
            prev_concepts = prev_training.get('concepts', 0)
            prev_hypotheses = prev_training.get('hypotheses', 0)
            
            concepts_gained = concepts - prev_concepts
            hypotheses_gained = hypotheses - prev_hypotheses
            
            if concepts_gained > 0 or hypotheses_gained > 0:
                print(f"\n📚 Learning Since Last Test:")
                print(f"   Concepts: +{concepts_gained:,}")
                print(f"   Hypotheses: +{hypotheses_gained:,}")
        
        print()
        print(f"🎯 Total Tests Completed: {len(test_history)}")
        print(f"📊 View detailed metrics: https://wandb.ai/erangross/true-agi-system/weave")
        
    except Exception as e:
        print(f"❌ Error reading test history: {e}")

if __name__ == "__main__":
    show_latest_progress()
