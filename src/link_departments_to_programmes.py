"""
link_departments_to_programmes.py
===================================
Bridges the two halves of the Neo4j graph by linking the existing
Faculty/Department nodes to the 24 Programme nodes that have HAS_UNIT data.

Source of truth: programs.csv (columns: name, faculty, department)

Strategy:
  1. Load programmes from programs.csv -> build {name: (faculty, dept)} lookup
  2. Fetch all Programme nodes from Neo4j (the 24 with HAS_UNIT)
  3. Fuzzy-match each Programme node name to a row in the CSV
  4. For each match, create:
       Department -[:HAS_PROGRAMME]-> Programme
       (The Faculty -> Department link already exists for the Department)
  5. Print a match report and ask for confirmation before writing

Result after running:
  Faculty ->[:FACULTY_HAS_DEPARTMENT]-> Department ->[:DEPARTMENT_OFFERS_PROGRAM]-> Program (252)
                                                   ->[:HAS_PROGRAMME]-> Programme(24) ->[:HAS_UNIT]-> ...
"""

import os
from difflib import SequenceMatcher
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv('d:/Jupyter notebook/Graph rag/.env')

BASE = 'd:/Jupyter notebook/Graph rag'
CSV_PATH = f'{BASE}/programs.csv'

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

DIVIDER = "=" * 60


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    # ── 1. Load CSVs and merge in-memory to bypass file locks ────────
    print("Loading data from programs_copy.csv and programs_new.csv...")
    try:
        import shutil
        # Copy programs.csv to avoid Windows Excel locks
        shutil.copy(f'{BASE}/programs.csv', f'{BASE}/programs_safe_read.csv')
        df_all = pd.read_csv(f'{BASE}/programs_safe_read.csv').fillna('')
    except Exception as e:
        print(f"Error loading programs: {e}")
        return

    try:
        df_dept = pd.read_csv(f'{BASE}/programs_new.csv').fillna('')
        lookup_dept = {}
        for _, row in df_dept.iterrows():
            nk = str(row['name']).strip().lower()
            if nk and row['department']:
                lookup_dept[nk] = row['department']
    except:
        lookup_dept = {}

    csv_lookup = []
    # Build {name: (faculty, dept)} lookup from df_all
    for _, row in df_all.iterrows():
        prog_name = str(row['name']).strip()
        if not prog_name: continue
        
        fac = str(row['faculty']).strip()
        dept = str(row.get('department', '')).strip()
        
        # Merge department from programs_new.csv if missing
        if prog_name.lower() in lookup_dept:
            dept = lookup_dept[prog_name.lower()]
        if not dept:
            dept = fac # fallback
            
        csv_lookup.append((prog_name, fac, dept))

    print(f"Loaded {len(csv_lookup)} programmes for matching.")

    # ── 2. Fetch Programme nodes from Neo4j ────────────────────────────
    print("Fetching Programme nodes from Neo4j...")
    with driver.session() as s:
        programme_rows = s.run("MATCH (p:Programme) RETURN p.name AS name").data()

    programmes = [r["name"] for r in programme_rows if r.get("name")]
    print(f"Found {len(programmes)} Programme nodes in Neo4j:\n")
    for p in programmes:
        print(f"  • {p}")

    # ── 3. Fuzzy-match each Programme to a CSV row ─────────────────────
    print(f"\n{DIVIDER}")
    print("MATCHING Programme nodes to programs.csv...")
    print(DIVIDER)

    THRESHOLD = 0.45
    matched = []
    unmatched = []

    for prog_name in programmes:
        best_score = 0
        best_csv_name = None
        best_faculty = None
        best_dept = None

        for csv_name, csv_fac, csv_dept in csv_lookup:
            score = similarity(prog_name, csv_name)
            if score > best_score:
                best_score = score
                best_csv_name = csv_name
                best_faculty = csv_fac
                best_dept = csv_dept

        if best_score >= THRESHOLD and best_faculty:
            matched.append({
                "programme": prog_name,
                "csv_name":  best_csv_name,
                "faculty":   best_faculty,
                "department": best_dept,
                "score":     round(best_score, 2),
            })
            print(f"  OK  [{best_score:.2f}] '{prog_name}'")
            print(f"         CSV match  : '{best_csv_name}'")
            print(f"         Department : {best_dept}")
        else:
            unmatched.append(prog_name)
            info = f"(best: '{best_csv_name}' @ {best_score:.2f})" if best_csv_name else ""
            print(f"  NO  '{prog_name}' — no match {info}")

    print(f"\nMatched  : {len(matched)}")
    print(f"Unmatched: {len(unmatched)}")

    if not matched:
        print("\nNo matches found. Check THRESHOLD or CSV programme names.")
        driver.close()
        return

    # 4. Confirm before writing
    print(f"\n{DIVIDER}")
    answer = input("Write Department -[:HAS_PROGRAMME]-> Programme to Neo4j? (yes/no): ").strip().lower()
    if answer != "yes":
        print("Aborted — no changes made.")
        driver.close()
        return

    # 5. Write relationships
    data = [{"programme": m["programme"], "department": m["department"]} for m in matched]

    with driver.session() as s:
        # Match department by checking if it contains the core name (e.g. "Computer Sciences")
        # because the CSV has "DEPARTMENT OF X" but Neo4j might have "Department of X" or just "X"
        result = s.run("""
        UNWIND $data AS row
        // Normalize: uppercase, replace 'AND' with '&' to handle variants
        WITH row,
             toUpper(row.department) AS dept_upper,
             toUpper(replace(row.department, ' AND ', ' & ')) AS dept_amp
        
        MATCH (org)
        WHERE (org:Department OR org:Faculty) 
          AND (
            toUpper(org.name) = dept_upper
            OR toUpper(org.name) = dept_amp
            OR dept_upper CONTAINS toUpper(org.name)
            OR toUpper(org.name) CONTAINS dept_upper
          )
        
        MATCH (prog:Programme {name: row.programme})
        MERGE (org)-[:HAS_PROGRAMME]->(prog)
        RETURN count(DISTINCT prog) AS created
        """, data=data)
        
        created = result.single()["created"]

        # Verify
        total = s.run("""
            MATCH (org)-[:HAS_PROGRAMME]->(p:Programme)
            RETURN count(*) AS n
        """).single()["n"]

    print(f"\nDone! {created} Programme nodes linked to a Department.")
    print(f"Total HAS_PROGRAMME relationships in graph: {total}")
    print("\nFull hierarchy now:")
    print("  Faculty -> Department -> Programme -> CourseUnit -> PastPaper")

    if unmatched:
        print(f"\nWarning: {len(unmatched)} Programme(s) not matched to a Department:")
        for u in unmatched:
            print(f"  - {u}")

    driver.close()


if __name__ == "__main__":
    main()
