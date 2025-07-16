#!/usr/bin/env python3
"""
Windows 11 Python Process Killer
Simple script to manually kill all Python processes when Ctrl+C doesn't work
"""

import subprocess
import psutil
import os
import time
import ctypes

def kill_all_python_processes():
    """Kill all Python processes on Windows 11 using multiple methods"""
    current_pid = os.getpid()
    print(f"[KILL] Current PID: {current_pid}")
    print("[KILL] 🔥 Killing all Python processes on Windows 11...")
    
    # Method 1: Taskkill with tree termination (most effective)
    print("[KILL] Method 1: Taskkill with process tree termination...")
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "python.exe", "/T"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15
        )
        print(f"[KILL] Taskkill result: {result.returncode}")
        if result.stdout:
            print(f"[KILL] Output: {result.stdout}")
        if result.stderr:
            print(f"[KILL] Stderr: {result.stderr}")
    except Exception as e:
        print(f"[KILL] Taskkill error: {e}")
    
    # Method 2: PowerShell process termination
    print("[KILL] Method 2: PowerShell process termination...")
    try:
        ps_cmd = f"Get-Process -Name 'python' -ErrorAction SilentlyContinue | Where-Object {{$_.Id -ne {current_pid}}} | Stop-Process -Force -ErrorAction SilentlyContinue"
        result = subprocess.run(
            ["powershell", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            check=False,
            timeout=15
        )
        print(f"[KILL] PowerShell result: {result.returncode}")
        if result.stderr:
            print(f"[KILL] PowerShell stderr: {result.stderr}")
    except Exception as e:
        print(f"[KILL] PowerShell error: {e}")
    
    # Method 3: Individual PID termination
    print("[KILL] Method 3: Individual PID termination...")
    try:
        python_processes = []
        for process in psutil.process_iter(['pid', 'name']):
            try:
                if (process.info['name'] == 'python.exe' and 
                    process.info['pid'] != current_pid):
                    python_processes.append(process.info['pid'])
            except:
                pass
        
        if python_processes:
            print(f"[KILL] Found {len(python_processes)} Python processes: {python_processes}")
            
            for pid in python_processes:
                # Try taskkill first
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True,
                        check=False,
                        timeout=5
                    )
                    print(f"[KILL] ✅ Taskkill killed PID {pid}")
                except Exception as e:
                    print(f"[KILL] Taskkill error for PID {pid}: {e}")
                
                # Try Windows API
                try:
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.OpenProcess(0x0001, False, pid)
                    if handle:
                        kernel32.TerminateProcess(handle, 1)
                        kernel32.CloseHandle(handle)
                        print(f"[KILL] ✅ Windows API killed PID {pid}")
                except Exception as e:
                    print(f"[KILL] Windows API error for PID {pid}: {e}")
        else:
            print("[KILL] No Python processes found to kill")
            
    except Exception as e:
        print(f"[KILL] Individual PID method error: {e}")
    
    # Method 4: WMIC (legacy but sometimes works)
    print("[KILL] Method 4: WMIC process termination...")
    try:
        result = subprocess.run(
            ["wmic", "process", "where", f"name='python.exe' and processid!={current_pid}", "delete"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15
        )
        print(f"[KILL] WMIC result: {result.returncode}")
        if result.stdout:
            print(f"[KILL] WMIC output: {result.stdout}")
    except Exception as e:
        print(f"[KILL] WMIC error: {e}")
    
    # Wait and verify
    print("[KILL] Waiting 3 seconds for processes to die...")
    time.sleep(3)
    
    # Final verification
    remaining = []
    try:
        for process in psutil.process_iter(['pid', 'name']):
            try:
                if (process.info['name'] == 'python.exe' and 
                    process.info['pid'] != current_pid):
                    remaining.append(process.info['pid'])
            except:
                pass
    except:
        pass
    
    if remaining:
        print(f"[KILL] ⚠️ {len(remaining)} Python processes still running: {remaining}")
        print("[KILL] 💡 You may need to:")
        print("[KILL]    1. Open Task Manager (Ctrl+Shift+Esc)")
        print("[KILL]    2. Find python.exe processes")
        print("[KILL]    3. Right-click -> End task")
        print("[KILL]    4. Or run as Administrator: taskkill /F /IM python.exe /T")
        return False
    else:
        print("[KILL] ✅ All Python processes successfully terminated!")
        return True

if __name__ == "__main__":
    print("Windows 11 Python Process Killer")
    print("================================")
    success = kill_all_python_processes()
    if success:
        print("\n🎉 SUCCESS: All Python processes killed!")
    else:
        print("\n⚠️  WARNING: Some processes may still be running")
        print("   Try running this script as Administrator")
    
    input("\nPress Enter to exit...")
