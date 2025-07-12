#!/usr/bin/env python3
"""
ICA Framework Main Launcher

Unified launcher for all ICA Framework functionality with organized file structure.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """Print the ICA Framework banner."""
    print("🧠" + "=" * 70)
    print("   INTEGRATED CAUSAL AGENCY (ICA) FRAMEWORK")
    print("   Advanced AGI with Intrinsic Curiosity & Causal Reasoning")
    print("=" * 72)

def print_menu():
    """Print the main menu."""
    print("\n🚀 Available Commands:")
    print("=" * 50)
    print("1. 🧪 Run Demo                    - examples/demo.py")
    print("2. 🔄 Continuous Learning         - scripts/continuous_learning.py")
    print("3. 📊 Monitor Learning            - scripts/monitor_continuous_learning.py")
    print("4. 🕸️  View Knowledge Graph       - scripts/view_knowledge_graph.py")
    print("5. 🧪 Run Tests                   - tests/test_components.py")
    print("6. 📋 Show Project Structure")
    print("7. 📦 Install Dependencies")
    print("8. ❌ Exit")
    print("=" * 50)

def run_demo():
    """Run the ICA framework demo."""
    print("\n🧪 Running ICA Framework Demo...")
    subprocess.run([sys.executable, "examples/demo.py"], cwd=project_root)

def run_continuous_learning():
    """Launch continuous learning with options."""
    print("\n🔄 Continuous Learning Options:")
    print("1. Quick test (100 steps)")
    print("2. Short session (1 hour)")
    print("3. Long session (8 hours)")
    print("4. Continuous (until stopped)")
    print("5. Custom configuration")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    cmd = [sys.executable, "scripts/continuous_learning.py"]
    
    if choice == "1":
        cmd.extend(["--max-steps", "100"])
    elif choice == "2":
        cmd.extend(["--max-hours", "1"])
    elif choice == "3":
        cmd.extend(["--max-hours", "8"])
    elif choice == "4":
        pass  # No limits
    elif choice == "5":
        max_steps = input("Max steps (or Enter for unlimited): ").strip()
        if max_steps:
            cmd.extend(["--max-steps", max_steps])
        max_hours = input("Max hours (or Enter for unlimited): ").strip()
        if max_hours:
            cmd.extend(["--max-hours", max_hours])
    else:
        print("Invalid choice")
        return
    
    # Update data directory path
    cmd.extend(["--save-dir", "data/continuous_learning_data"])
    
    print(f"\n🚀 Starting: {' '.join(cmd)}")
    print("Press Ctrl+C to stop learning...")
    subprocess.run(cmd, cwd=project_root)

def run_monitor():
    """Run the learning monitor."""
    print("\n📊 Starting Learning Monitor...")
    print("Choose monitoring mode:")
    print("1. Real-time dashboard")
    print("2. Text-only monitoring")
    print("3. Generate HTML dashboard")
    
    choice = input("Select option (1-3): ").strip()
    
    cmd = [sys.executable, "scripts/monitor_continuous_learning.py"]
    cmd.extend(["--data-dir", "data/continuous_learning_data"])
    
    if choice == "2":
        cmd.append("--text-only")
    elif choice == "3":
        cmd.append("--generate-dashboard")
    
    subprocess.run(cmd, cwd=project_root)

def view_knowledge_graph():
    """View the current knowledge graph."""
    print("\n🕸️ Knowledge Graph Viewer...")
    print("1. Visual graph (with plots)")
    print("2. Text analysis only")
    print("3. Detailed statistics")
    
    choice = input("Select option (1-3): ").strip()
    
    cmd = [sys.executable, "scripts/view_knowledge_graph.py"]
    cmd.extend(["--data-dir", "data/continuous_learning_data"])
    
    if choice == "2":
        cmd.append("--no-plot")
    elif choice == "3":
        cmd.extend(["--details", "--no-plot"])
    elif choice == "1":
        cmd.append("--details")
    
    subprocess.run(cmd, cwd=project_root)

def run_tests():
    """Run the test suite."""
    print("\n🧪 Running Test Suite...")
    subprocess.run([sys.executable, "tests/test_components.py"], cwd=project_root)

def show_project_structure():
    """Show the project directory structure."""
    print("\n📋 Project Structure:")
    print("=" * 50)
    
    structure = """
📁 ICA Framework/
├── 📁 ica_framework/          # Core framework code
│   ├── 📁 components/         # Main AI components
│   ├── 📁 core/              # ICA agent implementation
│   ├── 📁 sandbox/           # Testing environment
│   └── 📁 utils/             # Utilities and config
├── 📁 scripts/               # Execution scripts
│   ├── continuous_learning.py
│   ├── monitor_continuous_learning.py
│   └── view_knowledge_graph.py
├── 📁 examples/              # Demo and examples
├── 📁 tests/                 # Test suite
├── 📁 docs/                  # Documentation
├── 📁 config/                # Configuration files
├── 📁 data/                  # Generated data
│   ├── continuous_learning_data/
│   └── saved_agents/
├── 📁 logs/                  # Log files
├── 📁 requirements/          # Dependencies
│   ├── requirements.txt
│   ├── requirements-minimal.txt
│   └── requirements-optional.txt
└── 📁 results/               # Experimental results
"""
    print(structure)

def install_dependencies():
    """Install project dependencies."""
    print("\n📦 Installing Dependencies...")
    print("1. Minimal installation (core functionality)")
    print("2. Full installation (all features)")
    print("3. Optional packages (enhanced features)")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/requirements-minimal.txt"])
    elif choice == "2":
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/requirements.txt"])
    elif choice == "3":
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements/requirements-optional.txt"])
    else:
        print("Invalid choice")

def main():
    """Main launcher interface."""
    print_banner()
    
    while True:
        print_menu()
        choice = input("\n🎯 Enter your choice (1-8): ").strip()
        
        try:
            if choice == "1":
                run_demo()
            elif choice == "2":
                run_continuous_learning()
            elif choice == "3":
                run_monitor()
            elif choice == "4":
                view_knowledge_graph()
            elif choice == "5":
                run_tests()
            elif choice == "6":
                show_project_structure()
            elif choice == "7":
                install_dependencies()
            elif choice == "8":
                print("\n👋 Goodbye! Thanks for using ICA Framework!")
                break
            else:
                print("❌ Invalid choice. Please select 1-8.")
                
        except KeyboardInterrupt:
            print("\n\n⏸️ Operation cancelled by user.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
