#!/usr/bin/env python3
"""
ICA Framework Management Script
Comprehensive script to start, stop, monitor, and manage continuous learning
"""

import subprocess
import sys
import time
import os
from pathlib import Path
import argparse

class ICAManager:
    """Manages ICA Framework continuous learning processes"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.run_script = self.project_root / "run_continuous.py"
        self.log_dir = self.project_root / "logs"
        self.log_file = self.log_dir / "continuous_learning.log"
        
    def get_running_processes(self):
        """Get list of running continuous learning processes"""
        try:
            result = subprocess.run(
                ["powershell", "-Command", 
                 "Get-WmiObject Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_continuous.py*'} | Select-Object ProcessId, CommandLine, CreationDate"],
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                processes = []
                for line in lines:
                    if line.strip() and not line.strip().startswith('ProcessId') and not line.strip().startswith('-------'):
                        parts = line.strip().split()
                        if parts:
                            try:
                                pid = int(parts[0])
                                processes.append({"pid": pid, "line": line})
                            except ValueError:
                                continue
                return processes
            return []
        except Exception as e:
            print(f"[ERROR] Error checking processes: {e}")
            return []
    
    def start(self, force_restart=False):
        """Start continuous learning process (or restart if already running)"""
        print("=" * 60)
        print("ICA Framework - Starting Continuous Learning")
        print("=" * 60)
        
        # Check for existing processes
        existing = self.get_running_processes()
        if existing and not force_restart:
            print(f"[INFO] Found {len(existing)} process(es) already running:")
            for proc in existing:
                print(f"   PID {proc['pid']}")
            print("\n[INFO] Continuous learning is already active!")
            print("[TIP] Use 'Restart' option if you want to restart the process")
            return True
        elif existing and force_restart:
            print(f"[RESTART] Stopping {len(existing)} existing process(es)...")
            self.stop()
            time.sleep(2)  # Wait for processes to stop
        
        # Start new process
        if not self.run_script.exists():
            print(f"[ERROR] {self.run_script} not found!")
            return False
        
        print(f"[START] Starting continuous learning script: {self.run_script}")
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        try:
            print(f"[LOG] Output will be logged to: {self.log_file}")
            print("[INFO] Starting process in background...")
            
            with open(self.log_file, 'w') as log:
                process = subprocess.Popen(
                    [sys.executable, str(self.run_script)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    cwd=str(self.project_root)
                )
            
            print(f"[SUCCESS] Process started with PID: {process.pid}")
            print("[INFO] Running in background - use 'Status' or 'Monitor' to check progress")
            
            # Give process time to start
            time.sleep(3)
            
            if process.poll() is None:
                print("[OK] Process is running successfully!")
                self._show_initial_output()
                return True
            else:
                print("[ERROR] Process exited immediately!")
                self._show_error_output()
                return False
                
        except Exception as e:
            print(f"[ERROR] Failed to start process: {e}")
            return False
    
    def stop(self):
        """Stop all Python processes except this management script"""
        print("=" * 60)
        print("ICA Framework - Stopping All Python Processes")
        print("=" * 60)
        
        try:
            # Get current process PID to avoid killing ourselves
            current_pid = os.getpid()
            
            # Get all Python processes
            result = subprocess.run(
                ["powershell", "-Command", 
                 "Get-WmiObject Win32_Process | Where-Object {$_.Name -eq 'python.exe'} | Select-Object ProcessId, CommandLine"],
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0 or not result.stdout.strip():
                print("[INFO] No Python processes found.")
                return True
            
            lines = result.stdout.strip().split('\n')
            processes_to_kill = []
            
            for line in lines:
                if line.strip() and not line.strip().startswith('ProcessId') and not line.strip().startswith('-------'):
                    parts = line.strip().split()
                    if parts:
                        try:
                            pid = int(parts[0])
                            # Don't kill our own process
                            if pid != current_pid:
                                command_line = ' '.join(parts[1:]) if len(parts) > 1 else "unknown"
                                processes_to_kill.append({"pid": pid, "command": command_line})
                        except ValueError:
                            continue
            
            if not processes_to_kill:
                print("[INFO] No other Python processes found to stop.")
                return True
            
            print(f"[STOP] Found {len(processes_to_kill)} Python processes to stop:")
            for proc in processes_to_kill:
                # Show shortened command for readability
                short_cmd = proc['command'][:60] + "..." if len(proc['command']) > 60 else proc['command']
                print(f"   PID {proc['pid']}: {short_cmd}")
            
            # Kill all the processes
            killed_count = 0
            for proc in processes_to_kill:
                try:
                    subprocess.run(
                        ["powershell", "-Command", f"Stop-Process -Id {proc['pid']} -Force -ErrorAction SilentlyContinue"],
                        capture_output=True,
                        text=True
                    )
                    print(f"[STOPPED] Process {proc['pid']}")
                    killed_count += 1
                except Exception as e:
                    print(f"[ERROR] Failed to stop process {proc['pid']}: {e}")
            
            print(f"[SUCCESS] Stopped {killed_count} Python processes.")
            print(f"[INFO] Management script (PID {current_pid}) kept running.")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error stopping processes: {e}")
            return False
    
    def status(self):
        """Show simple status of continuous learning processes"""
        print("=" * 60)
        print("ICA Framework - Status Check")
        print("=" * 60)
        
        processes = self.get_running_processes()
        if processes:
            print(f"✅ RUNNING - {len(processes)} continuous learning process(es) active")
            for i, proc in enumerate(processes, 1):
                print(f"   Process {i}: PID {proc['pid']}")
            
            # Show basic runtime info only
            if self.log_file.exists():
                stat = self.log_file.stat()
                age_minutes = (time.time() - stat.st_mtime) / 60
                if age_minutes < 1:
                    runtime = "just started"
                elif age_minutes < 60:
                    runtime = f"~{int(age_minutes)} minutes"
                else:
                    runtime = f"~{int(age_minutes/60)} hours"
                print(f"   Runtime: {runtime}")
                print(f"   Log size: {stat.st_size:,} bytes")
        else:
            print("❌ STOPPED - No continuous learning processes running")
            print("   Use 'Start' option to begin learning")
        
        return len(processes) > 0
    
    def monitor(self):
        """Show recent log content instead of real-time monitoring"""
        if not self.log_file.exists():
            print("[ERROR] Log file not found. Make sure continuous learning is running.")
            print(f"[INFO] Expected location: {self.log_file}")
            return
        
        print("=" * 60)
        print("ICA Framework - Recent Log Output")
        print("=" * 60)
        
        try:
            stat = self.log_file.stat()
            print(f"Log file: {self.log_file.name}")
            print(f"Size: {stat.st_size:,} bytes")
            print(f"Last updated: {time.ctime(stat.st_mtime)}")
            
            if stat.st_size > 0:
                print(f"\n📝 Last 20 lines:")
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    recent_lines = lines[-20:] if len(lines) >= 20 else lines
                    if recent_lines:
                        for line in recent_lines:
                            clean_line = line.rstrip()
                            if clean_line:  # Only show non-empty lines
                                print(f"   {clean_line}")
                    else:
                        print("   (no content)")
            else:
                print("\n📝 No content logged yet")
                
        except Exception as e:
            print(f"[ERROR] Could not read log: {e}")
    
    def restart(self):
        """Restart continuous learning (stop then start)"""
        print("=" * 60)
        print("ICA Framework - Restarting Continuous Learning")
        print("=" * 60)
        
        existing = self.get_running_processes()
        if existing:
            print(f"[RESTART] Stopping {len(existing)} existing process(es)...")
            self.stop()
            time.sleep(2)
        else:
            print("[INFO] No existing processes found")
        
        print("[RESTART] Starting fresh process...")
        return self.start(force_restart=True)
    
    def _show_initial_output(self):
        """Show first few lines of log output"""
        if self.log_file.exists() and self.log_file.stat().st_size > 0:
            print("\n[LOG] Initial output:")
            try:
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    for line in lines[:8]:
                        print(f"   {line.rstrip()}")
                    if len(lines) > 8:
                        print(f"   ... (and {len(lines) - 8} more lines)")
            except Exception as e:
                print(f"   [ERROR] Could not read log: {e}")
    
    def _show_error_output(self):
        """Show error output from log"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if content.strip():
                        print("[ERROR] Process output:")
                        print(content)
                    else:
                        print("[ERROR] No output in log file")
            except Exception as e:
                print(f"[ERROR] Could not read log: {e}")

def show_menu():
    """Show interactive menu"""
    print("\n" + "=" * 50)
    print("🧠 ICA Framework Management")
    print("=" * 50)
    print()
    print("Choose an action:")
    print("  1. 🚀 Start     - Start continuous learning")
    print("  2. 🛑 Stop      - Stop all processes")  
    print("  3. 📊 Status    - Check current status")
    print("  4. � Restart   - Restart processes")
    print("  5. ❌ Exit      - Exit this menu")
    print()
    
def interactive_menu():
    """Run interactive menu loop"""
    manager = ICAManager()
    
    while True:
        show_menu()
        try:
            choice = input("Enter your choice (1-5): ").strip()
            print()  # Add spacing
            
            if choice == '1':
                # Start
                success = manager.start()
                if success:
                    print("\n✅ Process started successfully!")
                input("\nPress Enter to return to menu...")
                
            elif choice == '2':
                # Stop
                manager.stop()
                input("\nPress Enter to return to menu...")
                
            elif choice == '3':
                # Status
                running = manager.status()
                input("\nPress Enter to return to menu...")
                
            elif choice == '4':
                # Restart
                success = manager.restart()
                if success:
                    print("\n✅ Process restarted successfully!")
                input("\nPress Enter to return to menu...")
                
            elif choice == '5':
                # Exit
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            input("Press Enter to continue...")

def main():
    """Main function with command-line interface and interactive menu"""
    # If no arguments provided, show interactive menu
    if len(sys.argv) == 1:
        interactive_menu()
        return
    
    # Otherwise, use command-line interface
    parser = argparse.ArgumentParser(description="ICA Framework Management Tool")
    parser.add_argument('action', choices=['start', 'stop', 'status', 'monitor', 'restart'], 
                       help='Action to perform')
    
    args = parser.parse_args()
    manager = ICAManager()
    
    if args.action == 'start':
        success = manager.start()
        if success:
            print("\n[USAGE] Management commands:")
            print("   python scripts/manage_ica.py status    # Check status")
            print("   python scripts/manage_ica.py monitor   # Monitor real-time")
            print("   python scripts/manage_ica.py stop      # Stop processes")
            print("   python scripts/manage_ica.py restart   # Restart")
        sys.exit(0 if success else 1)
        
    elif args.action == 'stop':
        manager.stop()
        
    elif args.action == 'status':
        running = manager.status()
        if running:
            print("\n[USAGE] Monitor with: python scripts/manage_ica.py monitor")
        
    elif args.action == 'monitor':
        manager.monitor()
        
    elif args.action == 'restart':
        success = manager.restart()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
