#!/usr/bin/env python3
"""
TRUE AGI Continuous Learning Runner
Explores physical world simulation through curiosity-driven experimentation
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the TRUE AGI system
from scripts.components.core import AGIRunner

# Setup Windows encoding (W&B compatible)
if sys.platform == "win32":
    import codecs
    import locale
    
    # Set console code page to UTF-8
    os.system('chcp 65001 > nul')
    
    # Configure Python environment
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Set locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass  # Use system default
    
    # Skip stdout/stderr reconfiguration to avoid conflicts with W&B
    # W&B handles its own console capture and encoding


def main():
    """Main entry point - TRUE AGI through curiosity and experimentation"""
    print("🧠 TRUE AGI: Curiosity-Driven Physical World Explorer")
    print("=" * 60)
    
    # Create and run the TRUE AGI system
    runner = AGIRunner()
    runner.run()


if __name__ == "__main__":
    main()
