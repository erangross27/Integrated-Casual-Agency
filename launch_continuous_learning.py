#!/usr/bin/env python3
"""
Quick Launcher for Continuous Learning

This script provides easy commands to start continuous learning with different configurations.
"""

import os
import sys
import argparse
from pathlib import Path

def launch_continuous_learning():
    """Launch continuous learning with user-friendly options."""
    
    print("🧠 ICA Framework - Continuous Learning Launcher")
    print("=" * 60)
    
    # Get user preferences
    print("\nSelect learning mode:")
    print("1. Quick test (100 steps)")
    print("2. Short session (1 hour)")
    print("3. Long session (8 hours)")
    print("4. Continuous (until manually stopped)")
    print("5. Custom configuration")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    # Build command
    cmd_parts = [sys.executable, "continuous_learning.py"]
    
    if choice == "1":
        cmd_parts.extend(["--max-steps", "100"])
        print("\n🚀 Starting quick test (100 steps)...")
        
    elif choice == "2":
        cmd_parts.extend(["--max-hours", "1"])
        print("\n🚀 Starting short session (1 hour)...")
        
    elif choice == "3":
        cmd_parts.extend(["--max-hours", "8"])
        print("\n🚀 Starting long session (8 hours)...")
        
    elif choice == "4":
        print("\n🚀 Starting continuous learning (Ctrl+C to stop)...")
        
    elif choice == "5":
        max_steps = input("Maximum steps (or press Enter for unlimited): ").strip()
        if max_steps:
            cmd_parts.extend(["--max-steps", max_steps])
            
        max_hours = input("Maximum hours (or press Enter for unlimited): ").strip()
        if max_hours:
            cmd_parts.extend(["--max-hours", max_hours])
            
        save_dir = input("Save directory (or press Enter for default): ").strip()
        if save_dir:
            cmd_parts.extend(["--save-dir", save_dir])
            
        print(f"\n🚀 Starting custom session...")
        
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Show command and run
    print(f"Command: {' '.join(cmd_parts)}")
    print("\nPress Ctrl+C to stop the learning session at any time.")
    print("Progress will be automatically saved every 10 minutes.")
    print("-" * 60)
    
    try:
        os.execv(sys.executable, cmd_parts)
    except KeyboardInterrupt:
        print("\n\n🛑 Learning session stopped by user.")
    except Exception as e:
        print(f"\n❌ Error launching continuous learning: {e}")


if __name__ == "__main__":
    launch_continuous_learning()
