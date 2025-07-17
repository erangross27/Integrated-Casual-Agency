#!/usr/bin/env python3
"""
Modern ML Setup Script for ICA Framework
Sets up Weights & Biases and Weave for experiment tracking and observability
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_wandb():
    """Setup Weights & Biases for experiment tracking"""
    print("🚀 Setting up Weights & Biases (W&B) for TRUE AGI System...")
    print("\n📊 W&B provides:")
    print("- Real-time experiment tracking")
    print("- Neural network visualization")
    print("- GPU utilization monitoring")
    print("- Learning progress dashboards")
    print("- Model comparison and analysis")
    
    print("\n🔧 Setup Steps:")
    print("1. Create account at https://wandb.ai")
    print("2. Get your API key from https://wandb.ai/authorize")
    print("3. Run: wandb login")
    print("4. Your TRUE AGI dashboard will be at: https://wandb.ai/your-username/TRUE-AGI-System")
    
    input("\nPress Enter after creating your W&B account...")
    
    try:
        # Try to login to W&B
        result = subprocess.run(['wandb', 'login'], check=True, capture_output=True, text=True)
        print("✅ W&B login successful!")
    except subprocess.CalledProcessError:
        print("⚠️  Please run 'wandb login' manually after installation")
    except FileNotFoundError:
        print("⚠️  Please install wandb first: pip install wandb")

def setup_weave():
    """Setup Weave for function tracing"""
    print("\n🐝 Setting up Weave for Function Tracing...")
    print("\n🔍 Weave provides:")
    print("- Complete function input/output visibility")
    print("- AGI decision-making transparency")
    print("- Neural network prediction tracking")
    print("- Learning episode analysis")
    
    print("\n✅ Weave integrates automatically with your W&B project!")
    print("No additional setup required - just install the package.")

def check_modern_stack():
    """Check if modern ML stack is installed"""
    print("\n🔍 Checking Modern ML Stack Installation...")
    
    packages = {
        'wandb': 'Weights & Biases',
        'weave': 'Weave Function Tracing',
        'torch': 'PyTorch Neural Networks',
        'numpy': 'NumPy',
    }
    
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✅ {description}: Installed")
        except ImportError:
            print(f"❌ {description}: Not installed")
            print(f"   Install with: pip install {package}")

def main():
    """Main setup function"""
    print("🧠 TRUE AGI System - Modern ML Setup")
    print("=" * 50)
    
    print("\n🎯 This setup configures:")
    print("- Weights & Biases for experiment tracking")
    print("- Weave for function tracing")
    print("- File-based neural network storage")
    print("\n🚫 No PostgreSQL required! Pure ML-first architecture.")
    
    check_modern_stack()
    
    print("\n" + "="*50)
    setup_wandb()
    setup_weave()
    
    print("\n" + "="*50)
    print("🎉 Setup Complete!")
    print("\n🚀 Your TRUE AGI System now has:")
    print("- Industry-standard experiment tracking")
    print("- Complete observability")
    print("- Real-time dashboards")
    print("- Zero external database dependencies")
    
    print("\n📊 Once running, visit your dashboard:")
    print("https://wandb.ai/your-username/TRUE-AGI-System")

if __name__ == "__main__":
    main()
