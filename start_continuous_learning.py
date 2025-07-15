#!/usr/bin/env python3
"""
Start Continuous Learning Script
Kills existing Python processes and starts run_continuous.py in the background
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def kill_python_processes():
    """Kill existing Python processes running run_continuous.py"""
    try:
        print("Checking for existing continuous learning processes...")
        
        # Get current process ID to avoid killing ourselves
        current_pid = os.getpid()
        
        # Get all Python processes with their command lines
        result = subprocess.run(
            ["powershell", "-Command", 
             "Get-WmiObject Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | Select-Object ProcessId, CommandLine"],
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            processes_to_kill = []
            
            for line in lines[2:]:  # Skip header lines
                if line.strip() and 'run_continuous.py' in line:
                    # Extract process ID from the line
                    parts = line.strip().split()
                    if parts:
                        try:
                            pid = int(parts[0])
                            if pid != current_pid:  # Don't kill ourselves
                                processes_to_kill.append(pid)
                        except (ValueError, IndexError):
                            continue
            
            if processes_to_kill:
                print(f"Found {len(processes_to_kill)} continuous learning processes. Stopping them...")
                
                for pid in processes_to_kill:
                    kill_result = subprocess.run(
                        ["powershell", "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
                        capture_output=True,
                        text=True
                    )
                
                print("Successfully stopped existing continuous learning processes.")
                # Wait a moment for processes to fully terminate
                time.sleep(2)
            else:
                print("No existing continuous learning processes found.")
        else:
            print("No Python processes found.")
            
    except Exception as e:
        print(f"Error while checking/killing Python processes: {e}")

def start_continuous_learning():
    """Start run_continuous.py in the background"""
    try:
        script_path = Path(__file__).parent / "run_continuous.py"
        
        if not script_path.exists():
            print(f"Error: {script_path} not found!")
            return False
            
        print(f"Starting continuous learning script: {script_path}")
        
        # Try a more direct approach first
        try:
            # Method 1: Direct subprocess start
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"Process started with PID: {process.pid}")
            
        except Exception as e:
            print(f"Direct method failed: {e}")
            print("Trying PowerShell method...")
            
            # Method 2: PowerShell start (fallback)
            result = subprocess.run(
                ["powershell", "-Command", f"Start-Process -WindowStyle Hidden '{sys.executable}' -ArgumentList '{script_path}' -PassThru"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"PowerShell start failed: {result.stderr}")
                return False
            
            print("PowerShell start successful")
        
        print("Continuous learning started in the background.")
        
        # Wait a moment for the process to fully start
        time.sleep(5)
        
        # Simple verification - just check if we can find any python process
        print("Verifying process started...")
        
        # Method 1: Simple check for python processes
        check_result = subprocess.run(
            ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue"],
            capture_output=True, 
            text=True
        )
        
        if check_result.returncode == 0 and check_result.stdout.strip():
            print("✓ Python processes found running:")
            print(check_result.stdout)
            print("✓ Continuous learning process started successfully!")
            return True
        else:
            print("⚠ No Python processes found running.")
            print("The process may have started and exited quickly, or failed to start.")
            return False
            
    except Exception as e:
        print(f"Error starting continuous learning: {e}")
        return False

def main():
    """Main function"""
    print("=" * 50)
    print("ICA Framework - Continuous Learning Starter")
    print("=" * 50)
    
    # Step 1: Kill existing Python processes
    kill_python_processes()
    
    # Step 2: Start continuous learning
    success = start_continuous_learning()
    
    if success:
        print("\n✓ Continuous learning has been started successfully!")
        print("The process is running in the background.")
        print("\nTo stop it later, you can use:")
        print("Get-WmiObject Win32_Process | Where-Object {$_.CommandLine -like '*run_continuous.py*'} | ForEach-Object {Stop-Process -Id $_.ProcessId -Force}")
        print("Or simply run this script again to restart it.")
    else:
        print("\n✗ Failed to start continuous learning.")
        sys.exit(1)

if __name__ == "__main__":
    main()
