import sys
import pandas as pd
from neo4j_utils import get_driver, get_project_root, run_batch, close_driver, tc

ROOT = get_project_root()
CSV_PATH = ROOT / "data" / "curricular_mapping.csv"

# Alias mapping for unit-to-program link consistency
ALIASES = {
    "M.Ed.": "Master of Education",
    "B.Sc.": "Bachelor of Science",
    "B.A.": "Bachelor of Arts",
    "B.Com.": "Bachelor of Commerce",
}

MERGE_UNITS_QUERY = """
UNWIND $data AS row
MERGE (f:Faculty {name: row.faculty})
MERGE (d:Department {name: row.department})
MERGE (f)-[:HAS_DEPARTMENT]->(d)
MERGE (prog:Program {name: row.programme})
SET prog.units_ingested = true
MERGE (d)-[:OFFERS_PROGRAM]->(prog)
MERGE (u:CourseUnit {code: row.course_code})
SET u.name = row.course_name,
    u.is_current = row.is_current
MERGE (prog)-[:HAS_UNIT {year: row.year, semester: row.semester}]->(u)
"""

def normalize_prog(name):
    for alias, full in ALIASES.items():
        if alias.lower() in name.lower():
            return name.replace(alias, full)
    return name

def load_data():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        code = str(r.get("course_code", "")).strip().upper()
        # Resolve common program aliases
        raw_p = str(r.get("programme", "")).strip()
        programme = tc(normalize_prog(raw_p))
        faculty = tc(str(r.get("faculty", "")).strip())
        dept = tc(str(r.get("department", "")).strip())
        
        if not code or not programme or not faculty:
            continue
            
        # Refined hierarchy logic: only fallback to General if truly empty
        if not dept or dept.lower() in ["nan", "none", ""]:
            dept = f"{faculty} (General)"
            
        # Normalization for "Postgraduate Diploma Section"
        if faculty.lower() == "postgraduate diploma section":
            faculty = "Faculty of Education and Resources Development"

        rows.append({
            "course_code": code,
            "course_name": tc(str(r.get("course_name", "")).strip()),
            "programme": programme,
            "faculty": faculty,
            "department": dept,
            "year": str(r.get("year", "")).strip(),
            "semester": str(r.get("semester", "")).strip(),
            "is_current": str(r.get("is_jan_april_2026_offering", "True")).strip().lower() in ["true", "1", "1.0"],
        })

    # Unique rows by unit-program-year-semester
    seen, deduped = set(), []
    for r in rows:
        key = (r["course_code"], r["programme"], r["year"], r["semester"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    print(f"Loaded {len(deduped)} unit rows")
    return deduped

def main(dry_run=False):
    print("INGEST UNITS")
    rows = load_data()
    if dry_run:
        print(f"[DRY RUN] Would ingest {len(rows)} unit rows")
        return

    driver = get_driver()
    try:
        with driver.session() as session:
            run_batch(session, MERGE_UNITS_QUERY, rows, batch_size=500, label="unit rows")
        print("\nDone!")
    finally:
        close_driver(driver)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    main(dry_run=p.parse_args().dry_run)
