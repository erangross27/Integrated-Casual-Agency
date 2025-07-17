#!/usr/bin/env python3
"""
System Utilities Module
Handles configuration loading, signal handling, and system utilities
"""

import os
import sys
import json
import signal
import ctypes
import subprocess
from pathlib import Path


class SystemUtils:
    """Handles system utilities and configuration"""
    
    @staticmethod
    def setup_windows_encoding():
        """Setup Windows console encoding for Unicode support"""
        if sys.platform == "win32":
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
            os.system('chcp 65001 > nul')
            os.environ['PYTHONUNBUFFERED'] = '1'
    
    @staticmethod
    def flush_print(*args, **kwargs):
        """Print with immediate flush to ensure output appears"""
        print(*args, **kwargs)
        sys.stdout.flush()
    
    @staticmethod
    def load_database_config(project_root):
        """Load PostgreSQL configuration from config file"""
        config_file = project_root / "config/database/database_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    db_config = config_data['database']
                    print(f"[OK] Loaded PostgreSQL config from {config_file}")
                    print(f"   Host: {db_config['host']}:{db_config['port']}")
                    print(f"   Database: {db_config['database']}")
                    print(f"   User: {db_config['user']}")
                    print()
                    return db_config
                    
            except Exception as e:
                print(f"[ERROR] Failed to load config file: {e}")
                
        print("[WARNING] Using default PostgreSQL configuration")
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'ica_neural',
            'user': 'ica_user',
            'password': 'ica_password'
        }


class SignalHandler:
    """Handles system signals for graceful shutdown"""
    
    def __init__(self, shutdown_callback):
        self.shutdown_callback = shutdown_callback
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Windows-specific console control handler
        if sys.platform == "win32":
            self._setup_windows_console_handler()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n[STOP] 🛑 Received signal {signum}, shutting down gracefully...")
        print(f"[STOP] 💾 Ensuring neural network weights are saved...")
        
        if self.shutdown_callback:
            try:
                self.shutdown_callback()
            except Exception as e:
                print(f"[STOP] ⚠️ Error during shutdown: {e}")
        
        print(f"[STOP] ✅ Graceful shutdown complete")
        os._exit(0)
    
    def _setup_windows_console_handler(self):
        """Setup Windows console control handler"""
        try:
            from ctypes import wintypes
            
            def windows_console_handler(dwCtrlType):
                if dwCtrlType in (0, 1, 2):  # CTRL_C_EVENT, CTRL_BREAK_EVENT, CTRL_CLOSE_EVENT
                    print(f"\n[STOP] 🛑 Windows console event {dwCtrlType}, shutting down...")
                    if self.shutdown_callback:
                        self.shutdown_callback()
                    os._exit(0)
                return True
            
            # Register the console control handler
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCtrlHandler(
                ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(windows_console_handler),
                True
            )
            print("[INIT] ✅ Windows console control handler registered")
        except Exception as e:
            print(f"[INIT] Windows console handler setup failed: {e}")


class ProcessManager:
    """Manages process cleanup and termination"""
    
    @staticmethod
    def cleanup_processes():
        """Clean up all Python processes"""
        print("[CLEANUP] Starting process cleanup...")
        
        try:
            # Use taskkill to terminate Python processes
            result = subprocess.run(
                ["taskkill", "/F", "/IM", "python.exe", "/T"], 
                capture_output=True, 
                text=True,
                check=False,
                timeout=15
            )
            print(f"[CLEANUP] Process cleanup result: {result.returncode}")
        except Exception as e:
            print(f"[CLEANUP] Process cleanup error: {e}")
        
        print("[CLEANUP] ✅ Process cleanup completed")
