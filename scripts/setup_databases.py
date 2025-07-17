#!/usr/bin/env python3
"""
Database Setup Script for ICA Framework
Sets up PostgreSQL for neural network storage
"""

import os
import sys
import subprocess
from pathlib import Path

def install_postgresql():
    """Install PostgreSQL on Windows"""
    print("📦 Installing PostgreSQL...")
    print("✅ Great! You're installing PostgreSQL on Windows.")
    print("\nDuring installation, make sure to:")
    print("- Keep default username: postgres")
    print("- Remember the password you set for postgres user")
    print("- Keep default port: 5432")
    print("- Install pgAdmin (management tool)")
    print("- Install Stack Builder (optional)")
    
    input("\nPress Enter after PostgreSQL installation completes...")
    
    print("\n🔧 After installation, we'll help you:")
    print("1. Connect to PostgreSQL with your password")
    print("2. Create the ICA neural database")
    print("3. Create the ICA user account")
    print("4. Test the connection")

def create_database():
    """Create database and user for ICA neural storage"""
    print("🗄️ Creating ICA neural database...")
    
    # Get postgres password from user
    postgres_password = input("Enter the password you set for 'postgres' user during installation: ")
    
    print("\n🔧 Creating database and user...")
    
    # Test connection first
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='postgres',  # Connect to default database first
            user='postgres',
            password=postgres_password
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("✅ Connected to PostgreSQL successfully")
        
        # Create database
        try:
            cursor.execute("CREATE DATABASE ica_neural;")
            print("✅ Created database 'ica_neural'")
        except psycopg2.errors.DuplicateDatabase:
            print("ℹ️ Database 'ica_neural' already exists")
        
        # Create user
        try:
            cursor.execute("CREATE USER ica_user WITH PASSWORD 'ica_password';")
            print("✅ Created user 'ica_user'")
        except psycopg2.errors.DuplicateObject:
            print("ℹ️ User 'ica_user' already exists")
        
        # Grant privileges
        cursor.execute("GRANT ALL PRIVILEGES ON DATABASE ica_neural TO ica_user;")
        print("✅ Granted privileges to 'ica_user'")
        
        cursor.close()
        conn.close()
        
        print("\n✅ Database setup completed successfully!")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("\nManual setup instructions:")
        print("1. Open pgAdmin or psql")
        print("2. Connect as 'postgres' user")
        print("3. Run these commands:")
        print("   CREATE DATABASE ica_neural;")
        print("   CREATE USER ica_user WITH PASSWORD 'ica_password';")
        print("   GRANT ALL PRIVILEGES ON DATABASE ica_neural TO ica_user;")

def install_python_dependencies():
    """Install required Python packages"""
    print("🐍 Installing Python dependencies...")
    
    requirements = [
        "psycopg2-binary>=2.9.0"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ {req} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")

def test_connections():
    """Test database connections"""
    print("🔍 Testing PostgreSQL connection...")
    
    # Test PostgreSQL
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='ica_neural',
            user='ica_user',
            password='ica_password'
        )
        conn.close()
        print("✅ PostgreSQL connection successful")
        print("🧠 Neural network storage is ready!")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("Please ensure PostgreSQL is running and database is created")
        return False

def main():
    """Main setup process"""
    print("🚀 ICA Framework Database Setup")
    print("=" * 50)
    print("🧠 Setting up PostgreSQL-only architecture for TRUE AGI learning")
    print("💡 Neural networks ARE the knowledge - no graph database needed!")
    
    print("\n1. Installing Python dependencies...")
    install_python_dependencies()
    
    print("\n2. PostgreSQL Setup...")
    response = input("Have you finished installing PostgreSQL? (y/n): ").lower()
    if response == 'y':
        print("✅ Great! Let's configure the database...")
        create_database()
    else:
        print("📦 Please complete PostgreSQL installation first, then run this script again.")
        print("Download from: https://www.postgresql.org/download/windows/")
        return
    
    print("\n3. Testing connection...")
    if test_connections():
        print("\n✅ Setup complete!")
        print("\nYour ICA Framework now uses:")
        print("- 🐘 PostgreSQL for neural network storage (multi-GB efficient)")
        print("- 🧠 Neural networks store all learned knowledge")
        print("- 📊 Training metrics and learning progression")
        print("- 🎯 Environmental observations and responses")
        print("\nThis is a clean single-database architecture!")
        print("🚀 The AGI will learn from its surroundings and store knowledge in neural weights!")
    else:
        print("\n❌ Setup incomplete - please fix PostgreSQL connection")

if __name__ == "__main__":
    main()
