#!/usr/bin/env python3
"""
Quick restart script for ICA continuous learning
Automatically stops any existing process and starts fresh
"""

import os
import sys
import subprocess
import time
import signal

def kill_existing_processes():
    """Kill any existing Python processes running continuous learning"""
    try:
        if os.name == 'nt':  # Windows
            # Kill processes by name
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                         capture_output=True, check=False)
            subprocess.run(['taskkill', '/F', '/T', '/FI', 'WINDOWTITLE eq *continuous*'], 
                         capture_output=True, check=False)
        else:  # Unix/Linux
            subprocess.run(['pkill', '-f', 'run_continuous.py'], 
                         capture_output=True, check=False)
    except Exception:
        pass
    
    print("🔄 Stopped existing processes")
    time.sleep(2)

def start_learning():
    """Start the continuous learning process"""
    print("🚀 Starting optimized continuous learning...")
    print("✅ Aggressive logging suppression enabled")
    print("📊 Progress updates every 30 seconds")
    print("🔇 Worker spam eliminated")
    print("")
    
    # Run the continuous learning
    try:
        subprocess.run([sys.executable, 'run_continuous.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Learning stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🔧 ICA Framework - Quick Restart")
    print("=" * 50)
    
    kill_existing_processes()
    start_learning()
