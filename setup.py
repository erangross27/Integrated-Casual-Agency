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
    version="0.1.0",
    author="ICA Development Team",
    author_email="ica@example.com",
    description="Integrated Causal Agency Framework for Artificial General Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erangross27/Integrated-Casual-Agency",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "mypy>=1.4.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
            "plotly>=5.10.0",
            "pandas>=1.5.0",
            "tqdm>=4.60.0",
        ],
        "ml": [
            "pandas>=1.5.0",
            "pyro-ppl>=1.8.0",
            "botorch>=0.9.0",
            "gpytorch>=1.10.0",
            "stable-baselines3>=2.0.0",
            "gymnasium>=0.29.0",
            "torch-geometric>=2.0.0",
        ],
        "causal": [
            "causalnex>=0.12.0",
            "dowhy>=0.9.0",
            "pandas>=1.5.0",
        ],
        "monitoring": [
            "tqdm>=4.60.0",
            "wandb>=0.15.0",
            "python-dotenv>=1.0.0",
        ],
        "all": [
            # Development
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "mypy>=1.4.0",
            # Visualization
            "seaborn>=0.11.0", 
            "plotly>=5.10.0",
            "pandas>=1.5.0",
            "tqdm>=4.60.0",
            # Machine Learning
            "pyro-ppl>=1.8.0",
            "botorch>=0.9.0",
            "gpytorch>=1.10.0",
            "stable-baselines3>=2.0.0",
            "gymnasium>=0.29.0",
            "torch-geometric>=2.0.0",
            # Causal
            "causalnex>=0.12.0",
            "dowhy>=0.9.0",
            # Monitoring
            "wandb>=0.15.0",
            "python-dotenv>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ica-setup=setup:setup_database",
            "ica-learn=examples.learning:main",
        ]
    },
)


# Database setup functionality integrated into setup.py
def check_neo4j_availability():
    """Check if Neo4j driver is available"""
    try:
        import neo4j
        print("✅ Neo4j driver is available")
        return True
    except ImportError:
        print("❌ Neo4j driver not available")
        print("   Install with: pip install ica-framework[database]")
        return False


def setup_neo4j_config():
    """Interactive setup for Neo4j configuration"""
    print("\n🔧 Neo4j Configuration Setup")
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
            print(f"✅ Neo4j connection successful: {message}")
            driver.close()
            return True
            
    except ImportError:
        print("❌ Neo4j driver not installed")
        return False
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False


def create_sample_database_config():
    """Create sample database configuration files"""
    configs_dir = Path("config/database")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Neo4j config
    neo4j_config = {
        "backend": "neo4j", 
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
    
    print(f"✅ Neo4j configuration created in: {configs_dir}")
    print("💡 Edit neo4j.json with your actual Neo4j credentials")


def setup_database():
    """Interactive database setup"""
    print("🗄️ ICA Framework Database Setup")
    print("=" * 50)
    print("This helps you set up database backends for the ICA Framework")
    print()
    
    # Check current setup
    print("🔍 Checking current setup...")
    has_neo4j = check_neo4j_availability()
    
    # Main menu
    while True:
        print("\n📋 Available actions:")
        print("1. Create sample database configurations")
        print("2. Setup Neo4j configuration")
        print("3. Test Neo4j connection")
        print("4. Exit")
        
        choice = input("\nSelect an action (1-4): ").strip()
        
        if choice == '1':
            create_sample_database_config()
        
        elif choice == '2':
            config = setup_neo4j_config()
            
            # Save configuration
            configs_dir = Path("config/database")
            configs_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(configs_dir / "neo4j.json", 'w') as f:
                json.dump({"backend": "neo4j", "config": config}, f, indent=2)
            
            print(f"💾 Configuration saved to: {configs_dir / 'neo4j.json'}")
        
        elif choice == '3':
            if not has_neo4j:
                print("❌ Neo4j driver not available. Install with: pip install ica-framework[database]")
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
            print("👋 Database setup complete!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    # If run directly, do database setup
    if len(sys.argv) > 1 and sys.argv[1] == "database":
        setup_database()
    else:
        print("📦 ICA Framework Setup")
        print("=" * 30)
        print("Use: python setup.py database  # for database configuration")
        print("Use: pip install -e .         # for package installation")
        print("Use: pip install -e .[all]    # for all optional dependencies")
