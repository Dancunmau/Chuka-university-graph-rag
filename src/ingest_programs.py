import sys
import pandas as pd
from neo4j_utils import get_driver, get_project_root, run_batch, close_driver, tc

ROOT = get_project_root()
PROGRAMS_CSV = ROOT / "data" / "programs_linked.csv"

# Ingest Faculty -> Department -> Program hierarchy
MERGE_QUERY = """
UNWIND $data AS row
MERGE (f:Faculty {name: row.faculty})
MERGE (d:Department {name: row.department})
MERGE (f)-[:HAS_DEPARTMENT]->(d)
MERGE (p:Program {name: row.name})
SET p.level = row.level,
    p.duration_string = row.duration,
    p.fee_string = row.fee,
    p.units_ingested = false
MERGE (d)-[:OFFERS_PROGRAM]->(p)
"""

def load_data():
    if not PROGRAMS_CSV.exists():
        print(f"ERROR: {PROGRAMS_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(PROGRAMS_CSV).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        name = tc(str(r.get("name", "")).strip())
        faculty = tc(str(r.get("faculty", "")).strip())
        dept = tc(str(r.get("department", "")).strip())
        
        if not name or not faculty:
            continue
            
        # Refined hierarchy logic: only fallback to General if truly empty
        if not dept or dept.lower() in ["nan", "none", ""]:
            dept = f"{faculty} (General)"
            
        # Normalization for "Postgraduate Diploma Section"
        if faculty.lower() == "postgraduate diploma section":
            faculty = "Faculty of Education and Resources Development"

        rows.append({
            "name": name,
            "faculty": faculty,
            "department": dept,
            "level": str(r.get("level", "")).strip(),
            "duration": str(r.get("duration", "")).strip(),
            "fee": str(r.get("fee", "")).strip(),
        })
    print(f"Loaded {len(rows)} programs from {PROGRAMS_CSV.name}")
    return rows

def main(dry_run=False):
    print("INGEST PROGRAMS")
    rows = load_data()
    if dry_run:
        print(f"[DRY RUN] Would ingest {len(rows)} programs")
        return

    driver = get_driver()
    try:
        with driver.session() as session:
            run_batch(session, MERGE_QUERY, rows, batch_size=500, label="programs")
        print("\nDone!")
    finally:
        close_driver(driver)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    main(dry_run=p.parse_args().dry_run)
