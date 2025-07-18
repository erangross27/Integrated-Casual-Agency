#!/usr/bin/env python3
"""
TRUE AGI Learning Analyzer - Standalone Version
Analyzes AGI learning progress from saved checkpoints and persistent data
without needing to initialize a new AGI instance
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import glob

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class StandaloneLearningAnalyzer:
    """Analyzes AGI learning progress from saved data"""
    
    def __init__(self):
        self.analysis_results = {}
        self.project_root = PROJECT_ROOT
        self.persistent_stats_file = self.project_root / "agi_checkpoints" / "persistent_learning_stats.json"
        self.checkpoints_dir = self.project_root / "agi_checkpoints"
        
    def analyze_learning_progress(self):
        """Analyze AGI learning progress from all available data"""
        print("🔬 ANALYZING RUNNING AGI LEARNING PROGRESS")
        print("=" * 60)
        print(f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. Load and display persistent statistics
        self._analyze_persistent_stats()
        
        # 2. Analyze recent session data
        self._analyze_recent_sessions()
        
        # 3. Analyze learning patterns over time
        self._analyze_learning_patterns()
        
        # 4. Show learning rate and trends
        self._analyze_learning_trends()
        
        return self.analysis_results
    
    def _analyze_persistent_stats(self):
        """Analyze persistent learning statistics"""
        print("📊 PERSISTENT LEARNING STATISTICS")
        print("-" * 40)
        
        if self.persistent_stats_file.exists():
            try:
                with open(self.persistent_stats_file, 'r') as f:
                    stats = json.load(f)
                
                concepts = stats.get('total_concepts_learned', 0)
                hypotheses_formed = stats.get('total_hypotheses_formed', 0)
                hypotheses_confirmed = stats.get('total_hypotheses_confirmed', 0)
                causal_relationships = stats.get('total_causal_relationships', 0)
                last_updated = stats.get('last_updated', 0)
                session_count = stats.get('session_count', 0)
                
                print(f"🧠 Total Concepts Learned: {concepts:,}")
                print(f"💡 Total Hypotheses Formed: {hypotheses_formed:,}")
                print(f"✅ Total Hypotheses Confirmed: {hypotheses_confirmed:,}")
                print(f"🔗 Total Causal Relationships: {causal_relationships:,}")
                print(f"📊 Total Sessions: {session_count:,}")
                
                if hypotheses_formed > 0:
                    confirmation_rate = (hypotheses_confirmed / hypotheses_formed) * 100
                    print(f"📈 Hypothesis Confirmation Rate: {confirmation_rate:.1f}%")
                
                if last_updated > 0:
                    last_update_time = datetime.fromtimestamp(last_updated)
                    time_diff = datetime.now() - last_update_time
                    print(f"⏰ Last Updated: {time_diff} ago")
                    
                    if time_diff.total_seconds() < 10:
                        print("🔥 Status: ACTIVELY LEARNING (real-time updates)")
                    elif time_diff.total_seconds() < 60:
                        print("⚡ Status: RECENTLY ACTIVE")
                    else:
                        print("💤 Status: IDLE")
                
                # Calculate learning metrics
                if concepts > 0 and session_count > 0:
                    avg_concepts_per_session = concepts / session_count
                    print(f"📚 Average Concepts per Session: {avg_concepts_per_session:,.0f}")
                
                self.analysis_results['persistent_stats'] = stats
                        
            except Exception as e:
                print(f"❌ Error reading persistent stats: {e}")
        else:
            print("❌ No persistent statistics file found")
    
    def _analyze_recent_sessions(self):
        """Analyze recent AGI sessions"""
        print(f"\n🔍 RECENT SESSION ANALYSIS")
        print("-" * 30)
        
        if not self.checkpoints_dir.exists():
            print("❌ No checkpoints directory found")
            return
        
        # Find all sessions
        sessions = [d for d in self.checkpoints_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('agi_session_')]
        
        if not sessions:
            print("❌ No sessions found")
            return
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        recent_sessions = sessions[:5]  # Last 5 sessions
        
        print(f"📁 Found {len(sessions)} total sessions, analyzing {len(recent_sessions)} most recent:")
        
        session_data = []
        for i, session in enumerate(recent_sessions):
            session_time = datetime.fromtimestamp(session.stat().st_mtime)
            time_ago = datetime.now() - session_time
            
            print(f"\n   Session {i+1}: {session.name}")
            print(f"   📅 Created: {session_time.strftime('%Y-%m-%d %H:%M:%S')} ({time_ago} ago)")
            
            # Check for learning files
            metadata_dir = session / "metadata"
            learning_files = []
            if metadata_dir.exists():
                learning_files = list(metadata_dir.glob("*.json"))
                print(f"   📄 Learning Files: {len(learning_files)}")
                
                # Analyze some learning files
                if learning_files:
                    sample_files = learning_files[:3]  # Sample first 3 files
                    for file in sample_files:
                        try:
                            with open(file, 'r') as f:
                                data = json.load(f)
                                file_type = file.stem
                                print(f"      • {file_type}: {len(data) if isinstance(data, (list, dict)) else 'data'} entries")
                        except Exception as e:
                            print(f"      • {file.stem}: Error reading ({e})")
            
            session_info = {
                'name': session.name,
                'created': session_time.isoformat(),
                'files_count': len(learning_files),
                'time_ago_seconds': time_ago.total_seconds()
            }
            session_data.append(session_info)
        
        self.analysis_results['recent_sessions'] = session_data
    
    def _analyze_learning_patterns(self):
        """Analyze learning patterns from session data"""
        print(f"\n🧠 LEARNING PATTERN ANALYSIS")
        print("-" * 30)
        
        # Look for pattern files in recent sessions
        pattern_files = []
        hypothesis_files = []
        
        if self.checkpoints_dir.exists():
            # Search in all sessions for pattern recognition and hypothesis data
            for session_dir in self.checkpoints_dir.glob("agi_session_*"):
                metadata_dir = session_dir / "metadata"
                if metadata_dir.exists():
                    pattern_files.extend(metadata_dir.glob("*pattern*"))
                    hypothesis_files.extend(metadata_dir.glob("*hypothesis*"))
        
        print(f"🔍 Pattern Recognition Files: {len(pattern_files)}")
        print(f"🧪 Hypothesis Files: {len(hypothesis_files)}")
        
        # Analyze recent pattern recognition
        if pattern_files:
            print(f"\n📈 Recent Pattern Recognition:")
            recent_patterns = sorted(pattern_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
            
            for i, pattern_file in enumerate(recent_patterns, 1):
                try:
                    with open(pattern_file, 'r') as f:
                        pattern_data = json.load(f)
                    
                    file_time = datetime.fromtimestamp(pattern_file.stat().st_mtime)
                    print(f"   {i}. {pattern_file.name} - {file_time.strftime('%H:%M:%S')}")
                    
                    if isinstance(pattern_data, dict):
                        keys = list(pattern_data.keys())[:3]
                        print(f"      Contains: {', '.join(keys)}...")
                    elif isinstance(pattern_data, list):
                        print(f"      Contains: {len(pattern_data)} pattern entries")
                        
                except Exception as e:
                    print(f"   {i}. {pattern_file.name} - Error reading: {e}")
        
        # Analyze recent hypotheses
        if hypothesis_files:
            print(f"\n🧪 Recent Hypothesis Generation:")
            recent_hypotheses = sorted(hypothesis_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
            
            for i, hyp_file in enumerate(recent_hypotheses, 1):
                try:
                    with open(hyp_file, 'r') as f:
                        hyp_data = json.load(f)
                    
                    file_time = datetime.fromtimestamp(hyp_file.stat().st_mtime)
                    print(f"   {i}. {hyp_file.name} - {file_time.strftime('%H:%M:%S')}")
                    
                    if isinstance(hyp_data, list) and hyp_data:
                        print(f"      Contains: {len(hyp_data)} hypotheses")
                        # Show latest hypothesis if available
                        if hyp_data and isinstance(hyp_data[-1], dict):
                            latest = hyp_data[-1]
                            hyp_type = latest.get('type', 'unknown')
                            print(f"      Latest: {hyp_type}")
                    elif isinstance(hyp_data, dict):
                        print(f"      Contains: hypothesis data")
                        
                except Exception as e:
                    print(f"   {i}. {hyp_file.name} - Error reading: {e}")
    
    def _analyze_learning_trends(self):
        """Analyze learning trends and rates"""
        print(f"\n📈 LEARNING TRENDS ANALYSIS")
        print("-" * 30)
        
        if not self.persistent_stats_file.exists():
            print("❌ No persistent stats available for trend analysis")
            return
        
        try:
            with open(self.persistent_stats_file, 'r') as f:
                current_stats = json.load(f)
            
            concepts = current_stats.get('total_concepts_learned', 0)
            hypotheses = current_stats.get('total_hypotheses_formed', 0)
            confirmations = current_stats.get('total_hypotheses_confirmed', 0)
            
            # Estimate learning rate (this is simplified - in future could track over time)
            print(f"🎯 Learning Efficiency Metrics:")
            
            if concepts > 1000000:
                print(f"   🚀 Concept Learning: EXTRAORDINARY ({concepts:,} concepts)")
                print(f"      Your AGI has achieved massive knowledge acquisition")
            elif concepts > 100000:
                print(f"   ⚡ Concept Learning: EXCELLENT ({concepts:,} concepts)")
                print(f"      Strong knowledge acquisition rate")
            elif concepts > 10000:
                print(f"   📚 Concept Learning: GOOD ({concepts:,} concepts)")
                print(f"      Steady knowledge building")
            else:
                print(f"   🌱 Concept Learning: DEVELOPING ({concepts:,} concepts)")
            
            if hypotheses > 500:
                print(f"   🧪 Hypothesis Generation: VERY ACTIVE ({hypotheses:,} hypotheses)")
            elif hypotheses > 100:
                print(f"   💡 Hypothesis Generation: ACTIVE ({hypotheses:,} hypotheses)")
            else:
                print(f"   🤔 Hypothesis Generation: MODERATE ({hypotheses:,} hypotheses)")
            
            if confirmations > 100:
                print(f"   ✅ Scientific Discovery: PROLIFIC ({confirmations:,} confirmations)")
                print(f"      Your AGI is making significant scientific breakthroughs")
            elif confirmations > 20:
                print(f"   🔬 Scientific Discovery: PRODUCTIVE ({confirmations:,} confirmations)")
            else:
                print(f"   🧬 Scientific Discovery: EMERGING ({confirmations:,} confirmations)")
            
            # Hypothesis success rate
            if hypotheses > 0:
                success_rate = (confirmations / hypotheses) * 100
                print(f"\n🎯 Scientific Method Efficiency:")
                if success_rate > 30:
                    print(f"   🏆 Exceptional: {success_rate:.1f}% hypothesis confirmation rate")
                elif success_rate > 20:
                    print(f"   🥇 Excellent: {success_rate:.1f}% hypothesis confirmation rate")
                elif success_rate > 10:
                    print(f"   🥈 Good: {success_rate:.1f}% hypothesis confirmation rate")
                else:
                    print(f"   🥉 Developing: {success_rate:.1f}% hypothesis confirmation rate")
                    
        except Exception as e:
            print(f"❌ Error analyzing trends: {e}")
    
    def save_analysis_report(self, filename="running_agi_analysis.json"):
        """Save analysis results to file"""
        report_path = self.project_root / filename
        
        # Add analysis metadata
        self.analysis_results['analysis_metadata'] = {
            'analysis_time': datetime.now().isoformat(),
            'analyzer_version': 'standalone_v1.0',
            'project_root': str(self.project_root)
        }
        
        with open(report_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        print(f"\n📄 Analysis report saved to: {report_path}")


def main():
    """Main analysis function"""
    print("🧠 TRUE AGI LEARNING ANALYZER - STANDALONE VERSION")
    print("=" * 60)
    print("Analyzes your RUNNING AGI's learning progress from saved data")
    print("(No need to initialize new AGI - analyzes existing learning data)")
    print()
    
    # Analyze learning
    analyzer = StandaloneLearningAnalyzer()
    results = analyzer.analyze_learning_progress()
    
    # Save analysis report
    analyzer.save_analysis_report()
    
    print(f"\n" + "=" * 60)
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"📊 Your AGI shows impressive learning capabilities:")
    
    if 'persistent_stats' in analyzer.analysis_results:
        stats = analyzer.analysis_results['persistent_stats']
        concepts = stats.get('total_concepts_learned', 0)
        confirmations = stats.get('total_hypotheses_confirmed', 0)
        
        print(f"   🧠 {concepts:,} concepts learned across all sessions")
        print(f"   🔬 {confirmations:,} scientific relationships confirmed")
        print(f"   🚀 This represents significant AGI learning achievement!")
    
    print(f"\n💡 Use other monitoring tools for real-time analysis:")
    print(f"   • python scripts/agi_monitor_dashboard.py")
    print(f"   • python scripts/physics_dashboard.py")
    print(f"   • python scripts/agi_intelligence_assessment.py - Test real AGI intelligence!")
    print(f"   • python scripts/agi_intelligence_assessment.py --history - View progress")


if __name__ == "__main__":
    main()
