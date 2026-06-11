# Database Connection Manager & Adapter
# Supports Dual-Mode: SQLite (for local zero-dependency setup) and PostgreSQL (for production/Docker)

import os
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor

# Environment Variables
DATABASE_TYPE = os.environ.get("DATABASE_TYPE", "sqlite") # 'sqlite' or 'postgres'
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smarthouse_db")
SQLITE_DB_PATH = os.environ.get("SQLITE_DB_PATH", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "house_intelligence.db"
))

def get_db_connection():
    """Returns a raw connection to SQLite or PostgreSQL depending on configuration"""
    if DATABASE_TYPE.lower() == "postgres":
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
        # Enable foreign key support in SQLite
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

def execute_query(query, params=None, fetch="all"):
    """
    Executes a query and returns results as list of dictionaries.
    Handles cursor type difference between sqlite3 and psycopg2.
    """
    conn = get_db_connection()
    try:
        if DATABASE_TYPE.lower() == "postgres":
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or ())
                if fetch == "all":
                    results = cur.fetchall()
                    # Convert RealDictCursor output to normal dicts
                    return [dict(r) for r in results] if results else []
                elif fetch == "one":
                    result = cur.fetchone()
                    return dict(result) if result else None
                conn.commit()
        else:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query, params or ())
            if fetch == "all":
                results = cur.fetchall()
                return [dict(r) for r in results] if results else []
            elif fetch == "one":
                result = cur.fetchone()
                return dict(result) if result else None
            conn.commit()
    finally:
        conn.close()

def init_db(schema_path):
    """
    Loads DDL schema and initializes database tables.
    Adapts PostgreSQL DDL to SQLite syntax on the fly if needed.
    """
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at {schema_path}")
        
    with open(schema_path, "r") as f:
        ddl_script = f.read()
        
    conn = get_db_connection()
    try:
        if DATABASE_TYPE.lower() == "postgres":
            with conn.cursor() as cur:
                cur.execute(ddl_script)
            conn.commit()
            print("PostgreSQL Database initialized successfully.")
        else:
            # Adapt PostgreSQL DDL to SQLite
            sqlite_ddl = ddl_script
            sqlite_ddl = sqlite_ddl.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            sqlite_ddl = sqlite_ddl.replace("NUMERIC(9, 6)", "REAL")
            sqlite_ddl = sqlite_ddl.replace("NUMERIC(10, 2)", "REAL")
            sqlite_ddl = sqlite_ddl.replace("NUMERIC(15, 2)", "REAL")
            sqlite_ddl = sqlite_ddl.replace("NUMERIC(5, 2)", "REAL")
            sqlite_ddl = sqlite_ddl.replace("TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP", "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP")
            sqlite_ddl = sqlite_ddl.replace("DATE NOT NULL", "TEXT NOT NULL")
            
            # Split commands and execute individually in SQLite
            # To handle any composite statements cleanly
            cur = conn.cursor()
            cur.executescript(sqlite_ddl)
            conn.commit()
            print(f"SQLite Database initialized successfully at {SQLITE_DB_PATH}.")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    # Test initialization
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema = os.path.join(base_dir, "data", "schema.sql")
    init_db(schema)
