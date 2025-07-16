#!/usr/bin/env python3
"""
Main TRUE AGI Runner Module - LEGACY
This file is kept for backwards compatibility
Please use the new modular core components instead
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .core import AGIRunner

# Legacy compatibility wrapper
class TrueAGIRunner(AGIRunner):
    """Legacy wrapper for backwards compatibility"""
    
    def __init__(self):
        super().__init__()
        print("[LEGACY] ⚠️ Using legacy TrueAGIRunner - consider upgrading to core.AGIRunner")


def main():
    """Main entry point"""
    runner = TrueAGIRunner()
    runner.run()


if __name__ == "__main__":
    main()
