#!/usr/bin/env python3
"""
Monitor Continuous Learning Script
Shows the status and output of the continuous learning process
"""

import subprocess
import time
from pathlib import Path

def check_continuous_learning_status():
    """Check if continuous learning is running and show status"""
    print("=" * 50)
    print("ICA Framework - Continuous Learning Monitor")
    print("=" * 50)
    
    # Check for running processes
    try:
        result = subprocess.run(
            ["powershell", "-Command", 
             "Get-WmiObject Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_continuous.py*'} | Select-Object ProcessId, CommandLine, CreationDate"],
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            # Filter out header lines and empty lines
            actual_processes = []
            for line in lines:
                if line.strip() and not line.strip().startswith('ProcessId') and not line.strip().startswith('-------'):
                    actual_processes.append(line)
            
            if actual_processes:
                print("[OK] Continuous learning processes found:")
                for line in actual_processes:
                    print(f"   {line}")
                print()
            else:
                print("[ERROR] No continuous learning processes found.")
                return False
        else:
            print("[ERROR] No continuous learning processes found.")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error checking processes: {e}")
        return False
    
    # Check log file - path to parent directory (ICA root) and then to logs
    project_root = Path(__file__).parent.parent
    log_file = project_root / "logs" / "continuous_learning.log"
    
    if log_file.exists():
        print(f"[LOG] Log file found: {log_file}")
        try:
            stat = log_file.stat()
            print(f"   Size: {stat.st_size:,} bytes")
            print(f"   Modified: {time.ctime(stat.st_mtime)}")
            
            # Show last few lines
            print("\n[OUTPUT] Last 15 lines of output:")
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines[-15:]:
                    print(f"   {line.rstrip()}")
                    
        except Exception as e:
            print(f"   [ERROR] Could not read log file: {e}")
    else:
        print(f"[ERROR] No log file found at: {log_file}")
    
    return True

def tail_log_file():
    """Show real-time log output (like tail -f)"""
    project_root = Path(__file__).parent.parent
    log_file = project_root / "logs" / "continuous_learning.log"
    
    if not log_file.exists():
        print("[ERROR] Log file not found. Make sure continuous learning is running.")
        print(f"[INFO] Expected location: {log_file}")
        return
    
    print(f"\n[MONITOR] Monitoring log file: {log_file}")
    print("Press Ctrl+C to stop monitoring...")
    print("-" * 50)
    
    try:
        # Open file and seek to end
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(0, 2)  # Go to end of file
            
            while True:
                line = f.readline()
                if line:
                    print(line.rstrip())
                else:
                    time.sleep(0.5)  # Wait a bit before checking again
                    
    except KeyboardInterrupt:
        print("\n[STOP] Stopped monitoring.")
    except Exception as e:
        print(f"\n[ERROR] Error monitoring log: {e}")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--tail":
        tail_log_file()
    else:
        status_ok = check_continuous_learning_status()
        
        if status_ok:
            print("\n[OPTIONS] Available commands:")
            print("   python scripts/monitor_continuous_learning.py --tail    # Monitor real-time output")
            print("   python scripts/start_continuous_learning.py            # Restart if needed")

if __name__ == "__main__":
    main()
