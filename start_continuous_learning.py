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
        
        # Create log file path for output
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "continuous_learning.log"
        
        try:
            # Method 1: Start with visible console window and log output
            print(f"Starting process with output logged to: {log_file}")
            
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                    cwd=str(script_path.parent)
                )
            
            print(f"Process started with PID: {process.pid}")
            print(f"Console window opened - you can see the process running")
            print(f"Output is also being logged to: {log_file}")
            
            # Give the process time to start up
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                print("[OK] Process is running successfully!")
                
                # Show first few lines of output if available
                if log_file.exists() and log_file.stat().st_size > 0:
                    print("\n[LOG] Initial output:")
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            for line in lines[:10]:  # Show first 10 lines
                                print(f"   {line.rstrip()}")
                            if len(lines) > 10:
                                print(f"   ... (and {len(lines) - 10} more lines)")
                    except Exception as e:
                        print(f"   Could not read log file: {e}")
                
                return True
            else:
                print("[ERROR] Process exited immediately!")
                return_code = process.returncode
                print(f"   Exit code: {return_code}")
                
                # Show any error output
                if log_file.exists():
                    print("[LOG] Error output:")
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            error_content = f.read()
                            if error_content.strip():
                                print(error_content)
                            else:
                                print("   (no output in log file)")
                    except Exception as e:
                        print(f"   Could not read log file: {e}")
                
                return False
            
        except Exception as e:
            print(f"Failed to start process: {e}")
            print("Trying PowerShell method as fallback...")
            
            # Method 2: PowerShell start (fallback)
            cmd = f"Start-Process -FilePath '{sys.executable}' -ArgumentList '{script_path}' -WorkingDirectory '{script_path.parent}' -PassThru"
            result = subprocess.run(
                ["powershell", "-Command", cmd],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"PowerShell start failed: {result.stderr}")
                return False
            
            print("PowerShell start successful")
            print("[OK] Process started in new window")
            return True
            
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
        print("\n[OK] Continuous learning has been started successfully!")
        print("The process is running in a new console window.")
        print("[LOG] You can check the log file at: logs/continuous_learning.log")
        print("\nTo stop it later, you can:")
        print("1. Close the console window, or")
        print("2. Use: Get-WmiObject Win32_Process | Where-Object {$_.CommandLine -like '*run_continuous.py*'} | ForEach-Object {Stop-Process -Id $_.ProcessId -Force}")
        print("3. Or simply run this script again to restart it.")
        print("\n[MONITOR] To monitor progress, check the log file or console window.")
    else:
        print("\n[ERROR] Failed to start continuous learning.")
        sys.exit(1)

if __name__ == "__main__":
    main()
