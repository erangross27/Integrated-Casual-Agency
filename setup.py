"""
Setup script for ICA Framework
"""

from setuptools import setup, find_packages
import sys
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ica-framework",
    version="1.0.0",
    author="ICA Development Team",
    author_email="ica@example.com",
    description="Modular TRUE AGI System with Dynamic GPU Acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erangross27/Integrated-Casual-Agency",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
        "Topic :: Scientific/Engineering :: Machine Learning",
    ],
    python_requires=">=3.13",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "mypy>=1.4.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
            "nvidia-ml-py>=12.0.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.10.0",
            "pandas>=1.5.0",
            "tqdm>=4.60.0",
            "matplotlib>=3.5.0",
        ],
        "ml": [
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "torch-geometric>=2.0.0",
            "transformers>=4.20.0",
        ],
        "database": [
            "neo4j>=5.0.0",
            "py2neo>=2021.2.0",
        ],
        "monitoring": [
            "tqdm>=4.60.0",
            "psutil>=5.9.0",
            "python-dotenv>=1.0.0",
            "rich>=13.0.0",
        ],
        "all": [
            # Development
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "mypy>=1.4.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            # GPU
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
            "nvidia-ml-py>=12.0.0",
            # Visualization
            "seaborn>=0.11.0", 
            "plotly>=5.10.0",
            "pandas>=1.5.0",
            "tqdm>=4.60.0",
            "matplotlib>=3.5.0",
            # Machine Learning
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "torch-geometric>=2.0.0",
            "transformers>=4.20.0",
            # Database
            "neo4j>=5.0.0",
            "py2neo>=2021.2.0",
            # Monitoring
            "psutil>=5.9.0",
            "python-dotenv>=1.0.0",
            "rich>=13.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ica-setup=setup:setup_database",
            "ica-modular=scripts.run_continuous_modular:main",
            "ica-gpu-config=scripts.components.gpu.gpu_config:main",
            "ica-database=scripts.components.database.database_manager:main",
        ]
    },
)


# Database setup functionality integrated into setup.py
def check_neo4j_availability():
    """Check if Neo4j driver is available"""
    try:
        import neo4j
        print("Neo4j driver is available")
        return True
    except ImportError:
        print("Neo4j driver not available")
        print("   Install with: pip install ica-framework[database]")
        return False


def setup_neo4j_config():
    """Interactive setup for Neo4j configuration"""
    print("\nNeo4j Configuration Setup")
    print("=" * 40)
    
    config = {}
    
    # Get connection details
    config['uri'] = input("Neo4j URI (default: neo4j://127.0.0.1:7687): ").strip() or "neo4j://127.0.0.1:7687"
    config['username'] = input("Username (default: neo4j): ").strip() or "neo4j"
    config['password'] = input("Password: ").strip() or "password"
    config['database'] = input("Database name (default: neo4j): ").strip() or "neo4j"
    
    return config


def test_neo4j_connection(config):
    """Test Neo4j connection with given configuration"""
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(
            config['uri'],
            auth=(config['username'], config['password'])
        )
        
        with driver.session(database=config['database']) as session:
            result = session.run("RETURN 'Connection successful' AS message")
            message = result.single()['message']
            print(f"Neo4j connection successful: {message}")
            driver.close()
            return True
            
    except ImportError:
        print("Neo4j driver not installed")
        return False
    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        return False


def create_sample_database_config():
    """Create sample database configuration files"""
    configs_dir = Path("config/database")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Neo4j config
    neo4j_config = {
        "description": "Neo4j database configuration for TRUE AGI system",
        "config": {
            "uri": "neo4j://127.0.0.1:7687",
            "username": "neo4j",
            "password": "password",
            "database": "neo4j"
        }
    }
    
    import json
    
    with open(configs_dir / "neo4j.json", 'w') as f:
        json.dump(neo4j_config, f, indent=2)
    
    print(f"Neo4j configuration created in: {configs_dir}")
    print("Edit neo4j.json with your actual Neo4j credentials")


def check_gpu_setup():
    """Check GPU setup and availability"""
    print("\nChecking GPU Setup...")
    print("=" * 30)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ GPU Available: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            print(f"✅ GPU Count: {gpu_count}")
            return True
        else:
            print("❌ CUDA not available")
            print("   Install CUDA-enabled PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install with: pip install ica-framework[gpu]")
        return False


def verify_modular_components():
    """Verify all modular components are accessible"""
    print("\nVerifying Modular Components...")
    print("=" * 35)
    
    components = [
        ("Main Runner", "scripts.components.main_runner"),
        ("GPU Config", "scripts.components.gpu.gpu_config"),
        ("GPU Processor", "scripts.components.gpu.gpu_processor"),
        ("Database Manager", "scripts.components.database.database_manager"),
        ("AGI Monitor", "scripts.components.monitoring.agi_monitor"),
        ("System Utils", "scripts.components.system.system_utils"),
    ]
    
    all_good = True
    for name, module_path in components:
        try:
            __import__(module_path)
            print(f"✅ {name}: Available")
        except ImportError as e:
            print(f"❌ {name}: Not available - {e}")
            all_good = False
    
    return all_good


def setup_database():
    """Interactive database setup"""
    print("ICA Framework Modular System Setup")
    print("=" * 50)
    print("Setting up the Modular TRUE AGI System")
    print()
    
    # Check current setup
    print("Checking system requirements...")
    has_neo4j = check_neo4j_availability()
    has_gpu = check_gpu_setup()
    components_ok = verify_modular_components()
    
    # Main menu
    while True:
        print("\nAvailable actions:")
        print("1. Create sample database configurations")
        print("2. Setup Neo4j configuration")
        print("3. Test Neo4j connection")
        print("4. Check GPU setup")
        print("5. Verify modular components")
        print("6. Run modular system test")
        print("7. Exit")
        
        choice = input("\nSelect an action (1-7): ").strip()
        
        if choice == '1':
            create_sample_database_config()
        
        elif choice == '2':
            config = setup_neo4j_config()
            
            # Save configuration
            configs_dir = Path("config/database")
            configs_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(configs_dir / "neo4j.json", 'w') as f:
                json.dump({"description": "Neo4j database configuration for TRUE AGI system", "config": config}, f, indent=2)
            
            print(f"Configuration saved to: {configs_dir / 'neo4j.json'}")
        
        elif choice == '3':
            if not has_neo4j:
                print("Neo4j driver not available. Install with: pip install ica-framework[database]")
                continue
            
            # Load configuration
            configs_dir = Path("config/database")
            config_file = configs_dir / "neo4j.json"
            
            if config_file.exists():
                import json
                with open(config_file, 'r') as f:
                    full_config = json.load(f)
                    config = full_config['config']
            else:
                config = setup_neo4j_config()
            
            test_neo4j_connection(config)
        
        elif choice == '4':
            check_gpu_setup()
        
        elif choice == '5':
            verify_modular_components()
        
        elif choice == '6':
            print("Testing modular system...")
            if components_ok and has_gpu:
                print("✅ All components ready - you can run: python scripts/run_continuous_modular.py")
            else:
                print("❌ Some components missing - check installation")
        
        elif choice == '7':
            print("Modular system setup complete!")
            break
        
        else:
            print("Invalid choice. Please select 1-7.")


if __name__ == "__main__":
    # If run directly, do system setup
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_database()
    else:
        print("ICA Framework - Modular TRUE AGI System Setup")
        print("=" * 50)
        print("Installation commands:")
        print("  pip install -e .              # Basic installation")
        print("  pip install -e .[gpu]         # With GPU support")
        print("  pip install -e .[database]    # With database support")
        print("  pip install -e .[all]         # Full installation")
        print()
        print("Setup commands:")
        print("  python setup.py setup         # Interactive system setup")
        print("  ica-setup                     # Setup via console command")
        print("  ica-modular                   # Run modular system")
        print()
        print("System requirements:")
        print("  - Python 3.13+")
        print("  - NVIDIA GPU with 4GB+ VRAM")
        print("  - Neo4j Database")
        print("  - CUDA Toolkit 11.8+ or 12.x")
