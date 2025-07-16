#!/usr/bin/env python3
"""
Simple TRUE AGI Continuous Learning Runner
Uses modular components for clean, maintainable code
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the modular TRUE AGI system
from components.main_runner import TrueAGIRunner

# Setup Windows encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    os.system('chcp 65001 > nul')
    os.environ['PYTHONUNBUFFERED'] = '1'


def main():
    """Main entry point - clean and simple"""
    print("🧠 TRUE AGI Continuous Learning System - Modular Edition")
    print("=" * 60)
    
    # Create and run the TRUE AGI system
    runner = TrueAGIRunner()
    runner.run()


if __name__ == "__main__":
    main()
