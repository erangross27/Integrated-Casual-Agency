#!/usr/bin/env python3
"""
Fix PostgreSQL permissions for ICA user
"""

import psycopg2

def fix_permissions():
    """Grant proper permissions to ica_user"""
    print("🔧 Fixing PostgreSQL permissions for ica_user...")
    
    # Get postgres password from user
    postgres_password = input("Enter the password for 'postgres' user: ")
    
    try:
        # Connect as postgres (admin)
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='ica_neural',
            user='postgres',
            password=postgres_password
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("✅ Connected as postgres user")
        
        # Grant schema permissions
        cursor.execute("GRANT CREATE ON SCHEMA public TO ica_user;")
        print("✅ Granted CREATE permission on schema public")
        
        cursor.execute("GRANT USAGE ON SCHEMA public TO ica_user;")
        print("✅ Granted USAGE permission on schema public")
        
        # Grant default privileges
        cursor.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ica_user;")
        print("✅ Granted default table privileges")
        
        cursor.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ica_user;")
        print("✅ Granted default sequence privileges")
        
        # Make ica_user owner of the database
        cursor.execute("ALTER DATABASE ica_neural OWNER TO ica_user;")
        print("✅ Made ica_user owner of ica_neural database")
        
        cursor.close()
        conn.close()
        
        print("\n✅ Permissions fixed successfully!")
        print("Now the TRUE AGI system can create tables automatically.")
        
    except Exception as e:
        print(f"❌ Failed to fix permissions: {e}")

if __name__ == "__main__":
    fix_permissions()
