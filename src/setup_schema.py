"""
setup_schema.py — Automatically apply Neo4j constraints from neo4j_schema.cypher.

Reads config/neo4j_schema.cypher and executes all CREATE CONSTRAINT statements.
Safe to run multiple times (uses IF NOT EXISTS).
"""

import sys
from neo4j_utils import get_driver, get_project_root, close_driver

ROOT = get_project_root()
SCHEMA_FILE = ROOT / "config" / "neo4j_schema.cypher"


def setup_constraints(driver):
    if not SCHEMA_FILE.exists():
        print(f"ERROR: Schema file not found at {SCHEMA_FILE}")
        sys.exit(1)

    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract all CREATE CONSTRAINT statements
    statements = [stmt.strip() for stmt in content.split(";") if "CREATE CONSTRAINT" in stmt.upper()]
    
    if not statements:
        print("No constraint statements found in the schema file.")
        return

    print(f"Applying {len(statements)} constraints to Neo4j database...")
    
    applied = 0
    with driver.session() as session:
        for stmt in statements:
            if not stmt:
                continue
            try:
                session.run(stmt)
                applied += 1
            except Exception as e:
                print(f"  Warning: Failed to apply constraint: {stmt}")
                print(f"  Error: {e}")
                
    print(f"Successfully ensured {applied} uniqueness constraints.")


def main(dry_run: bool = False):
    print("=" * 60)
    print("SETUP GRAPH SCHEMA  (Constraints)")
    print("=" * 60)
    
    if dry_run:
        print("[DRY RUN] Would check and apply database constraints.")
        return
    
    driver = get_driver()
    try:
        setup_constraints(driver)
        print("\nDone!")
    finally:
        close_driver(driver)


if __name__ == "__main__":
    main()
