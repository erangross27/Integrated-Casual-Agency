#!/usr/bin/env python3
"""
AGI Progress Monitoring Dashboard
Combines all monitoring tools to track AGI learning progress over time
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class AGIMonitorDashboard:
    """Comprehensive AGI monitoring dashboard"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.persistent_stats_file = self.project_root / "agi_checkpoints" / "persistent_learning_stats.json"
        
    def show_complete_status(self):
        """Show complete AGI status and progress"""
        print("🤖 TRUE AGI MONITORING DASHBOARD")
        print("=" * 60)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. System Status
        self._show_system_status()
        
        # 2. Persistent Learning Statistics
        self._show_persistent_stats()
        
        # 3. Recent Activity
        self._show_recent_activity()
        
        # 4. Learning Rate Analysis
        self._show_learning_rate()
        
        print("\n" + "=" * 60)
        print("💡 Use these scripts for detailed analysis:")
        print("   • python scripts/learning_analyzer.py - Detailed learning analysis")
        print("   • python scripts/physics_dashboard.py - Physics discoveries")
        print("   • python scripts/live_agi_check.py - Live system check")
        print("   • python scripts/simple_intelligence_test.py - Test AGI knowledge")
        print("   • python scripts/agi_intelligence_assessment.py - REAL AGI intelligence test!")
        print("   • python scripts/standalone_learning_analyzer.py - Learning progress analysis")
        
    def _show_system_status(self):
        """Check if AGI is currently running"""
        print("🎯 SYSTEM STATUS")
        print("-" * 20)
        
        try:
            # Use a more comprehensive approach to find Python processes
            result = subprocess.run([
                'powershell', '-Command', 
                'Get-WmiObject Win32_Process | Where-Object {$_.Name -eq "python.exe"} | Select-Object ProcessId, CommandLine'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                # Look for AGI-related scripts in the command lines
                agi_keywords = ["run_continuous", "demo.py", "agi", "physics", "learning"]
                running_processes = []
                
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'CommandLine' in line and any(keyword in line.lower() for keyword in agi_keywords):
                        # Extract script name from command line
                        if "run_continuous" in line.lower():
                            running_processes.append("run_continuous.py")
                        elif "demo.py" in line.lower():
                            running_processes.append("demo.py")
                        elif any(word in line.lower() for word in ["agi", "physics", "learning"]):
                            running_processes.append("AGI-related script")
                
                if running_processes:
                    print("✅ AGI System: RUNNING")
                    for process in set(running_processes):  # Remove duplicates
                        print(f"   • {process}")
                    return
            
            # Alternative method: Check for recent activity in persistent stats
            if self.persistent_stats_file.exists():
                try:
                    with open(self.persistent_stats_file, 'r') as f:
                        stats = json.load(f)
                    
                    last_updated = stats.get('last_updated', 0)
                    if last_updated > 0:
                        last_update_time = datetime.fromtimestamp(last_updated)
                        time_diff = datetime.now() - last_update_time
                        
                        if time_diff.total_seconds() < 5:  # Updated within last 5 seconds
                            print("✅ AGI System: CONFIRMED RUNNING")
                            print("   • Statistics updating in real-time")
                            print("   • AGI is actively learning")
                            return
                        elif time_diff.total_seconds() < 30:  # Updated within last 30 seconds
                            print("✅ AGI System: LIKELY RUNNING")
                            print("   • Statistics recently updated")
                            print("   • AGI appears active")
                            return
                except Exception:
                    pass
            
            # If we get here, no AGI processes found
            print("❌ AGI System: NOT DETECTED")
            print("   Note: AGI may be running but not detectable")
            print("   To start AGI:")
            print("   • python examples/demo.py")
            print("   • python scripts/run_continuous.py")
                
        except Exception as e:
            print(f"❓ AGI System: Cannot determine status")
            print(f"   Error: {e}")
            
            # Fallback: Check if stats are updating recently
            if self.persistent_stats_file.exists():
                try:
                    with open(self.persistent_stats_file, 'r') as f:
                        stats = json.load(f)
                    
                    last_updated = stats.get('last_updated', 0)
                    if last_updated > 0:
                        last_update_time = datetime.fromtimestamp(last_updated)
                        time_diff = datetime.now() - last_update_time
                        
                        if time_diff.total_seconds() < 30:  # Updated within last 30 seconds
                            print("   💡 But statistics are updating - AGI likely running!")
                except Exception:
                    pass
    
    def _show_persistent_stats(self):
        """Show persistent learning statistics"""
        print("\n📊 PERSISTENT LEARNING STATISTICS")
        print("-" * 35)
        
        if self.persistent_stats_file.exists():
            try:
                with open(self.persistent_stats_file, 'r') as f:
                    stats = json.load(f)
                
                concepts = stats.get('total_concepts_learned', 0)
                hypotheses_formed = stats.get('total_hypotheses_formed', 0)
                hypotheses_confirmed = stats.get('total_hypotheses_confirmed', 0)
                causal_relationships = stats.get('total_causal_relationships', 0)
                last_updated = stats.get('last_updated', 0)
                
                print(f"🧠 Concepts Learned: {concepts:,}")
                print(f"💡 Hypotheses Formed: {hypotheses_formed:,}")
                print(f"✅ Hypotheses Confirmed: {hypotheses_confirmed:,}")
                print(f"🔗 Causal Relationships: {causal_relationships:,}")
                
                if hypotheses_formed > 0:
                    confirmation_rate = (hypotheses_confirmed / hypotheses_formed) * 100
                    print(f"📈 Confirmation Rate: {confirmation_rate:.1f}%")
                
                if last_updated > 0:
                    last_update_time = datetime.fromtimestamp(last_updated)
                    time_diff = datetime.now() - last_update_time
                    print(f"⏰ Last Updated: {time_diff} ago")
                    
            except Exception as e:
                print(f"❌ Error reading persistent stats: {e}")
        else:
            print("❌ No persistent statistics file found")
    
    def _show_recent_activity(self):
        """Show recent AGI session activity"""
        print("\n🔍 RECENT SESSION ACTIVITY")
        print("-" * 25)
        
        checkpoints_dir = self.project_root / "agi_checkpoints"
        
        if checkpoints_dir.exists():
            # Find recent sessions
            sessions = [d for d in checkpoints_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('agi_session_')]
            
            if sessions:
                # Sort by creation time (newest first)
                sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                recent_sessions = sessions[:3]  # Last 3 sessions
                
                for i, session in enumerate(recent_sessions):
                    session_time = datetime.fromtimestamp(session.stat().st_mtime)
                    time_ago = datetime.now() - session_time
                    print(f"📁 Session {i+1}: {session.name}")
                    print(f"   Created: {time_ago} ago")
                    
                    # Check for learning files
                    metadata_dir = session / "metadata"
                    if metadata_dir.exists():
                        files = list(metadata_dir.glob("*.json"))
                        print(f"   Files: {len(files)} learning records")
                    print()
            else:
                print("❌ No recent sessions found")
        else:
            print("❌ No checkpoints directory found")
    
    def _show_learning_rate(self):
        """Analyze learning rate over time"""
        print("📈 LEARNING RATE ANALYSIS")
        print("-" * 25)
        
        # This is a simplified analysis - in the future you could track
        # statistics over time to show learning acceleration
        if self.persistent_stats_file.exists():
            try:
                with open(self.persistent_stats_file, 'r') as f:
                    stats = json.load(f)
                
                concepts = stats.get('total_concepts_learned', 0)
                
                # Rough estimation based on current totals
                if concepts > 1000000:
                    print("🚀 Learning Status: HIGHLY ACTIVE")
                    print("   Your AGI is processing massive amounts of information")
                elif concepts > 100000:
                    print("⚡ Learning Status: VERY ACTIVE") 
                    print("   Strong learning progression detected")
                elif concepts > 10000:
                    print("📚 Learning Status: ACTIVE")
                    print("   Steady learning progress")
                else:
                    print("🌱 Learning Status: STARTING")
                    print("   Early learning phase")
                    
            except Exception as e:
                print(f"❌ Error analyzing learning rate: {e}")


def main():
    """Main monitoring function"""
    dashboard = AGIMonitorDashboard()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        # Continuous monitoring mode
        print("🔄 Starting continuous monitoring (Ctrl+C to stop)")
        try:
            while True:
                dashboard.show_complete_status()
                print(f"\n⏰ Next update in 60 seconds...")
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
    else:
        # Single run
        dashboard.show_complete_status()


if __name__ == "__main__":
    main()
