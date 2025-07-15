# ICA Framework Management Scripts

This folder contains management scripts for the ICA Framework continuous learning system.

## Scripts Overview

### 1. `manage_ica.py` - **Main Management Tool** (Recommended)
Comprehensive script to manage all aspects of continuous learning.

**Usage:**
```bash
# Start continuous learning
python scripts/manage_ica.py start

# Check status
python scripts/manage_ica.py status

# Monitor real-time output
python scripts/manage_ica.py monitor

# Stop all processes
python scripts/manage_ica.py stop

# Restart (stop + start)
python scripts/manage_ica.py restart
```

### 2. `start_continuous_learning.py` - Legacy Starter
Starts the continuous learning process in a new console window.

**Usage:**
```bash
python scripts/start_continuous_learning.py
```

### 3. `monitor_continuous_learning.py` - Legacy Monitor
Monitors continuous learning processes and shows log output.

**Usage:**
```bash
# Check status
python scripts/monitor_continuous_learning.py

# Monitor real-time output
python scripts/monitor_continuous_learning.py --tail
```

## Quick Start

1. **Start learning:**
   ```bash
   python scripts/manage_ica.py start
   ```

2. **Check if it's working:**
   ```bash
   python scripts/manage_ica.py status
   ```

3. **Watch real-time progress:**
   ```bash
   python scripts/manage_ica.py monitor
   ```

4. **Stop when done:**
   ```bash
   python scripts/manage_ica.py stop
   ```

## Features

- ✅ **Unicode-safe**: All scripts handle Windows encoding properly
- ✅ **Process management**: Start, stop, restart processes safely
- ✅ **Real-time monitoring**: Watch progress as it happens
- ✅ **Log management**: Automatic logging to `logs/continuous_learning.log`
- ✅ **Error handling**: Graceful error reporting and recovery
- ✅ **Console windows**: Visual feedback with new console windows
- ✅ **Process detection**: Smart detection of running processes

## Log Files

All output is logged to: `logs/continuous_learning.log` (relative to project root)

## Troubleshooting

1. **Process won't start**: Check `logs/continuous_learning.log` for errors
2. **Unicode errors**: Scripts are now Unicode-safe for Windows
3. **Multiple processes**: Use `manage_ica.py stop` to clean up
4. **Permission issues**: Run as administrator if needed

## Migration from Root Scripts

The old scripts in the project root (`start_continuous_learning.py`, `monitor_continuous_learning.py`) can be safely deleted after testing these new scripts.

**Recommended workflow:**
1. Test the new scripts: `python scripts/manage_ica.py start`
2. Verify they work: `python scripts/manage_ica.py status`
3. Delete old scripts from project root (optional)
4. Update any documentation/shortcuts to use `scripts/manage_ica.py`
