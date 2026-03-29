import sys
import pandas as pd
from neo4j_utils import get_driver, get_project_root, run_batch, close_driver, tc

ROOT = get_project_root()
CSV_PATH = ROOT / "data" / "chuka_exam_papers_CLEANED.csv"

# Ingest PastPaper nodes
PAPER_MERGE_QUERY = """
UNWIND $data AS row
MERGE (pp:PastPaper {link: row.link})
SET pp.title       = row.title,
    pp.year        = row.year,
    pp.course_code = row.course_code
"""

# Link PastPaper -> CourseUnit (MERGE ensures papers are linked even if unit nodes don't exist yet)
LINK_QUERY = """
UNWIND $data AS row
MATCH (pp:PastPaper {link: row.link})
MERGE (u:CourseUnit {code: row.course_code})
MERGE (u)-[:HAS_PAST_PAPER]->(pp)
"""

def load_data():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={"repository_link": "link"}, inplace=True, errors="ignore")

    df = df.drop_duplicates(subset=["link"])
    
    all_papers = []
    for _, r in df.iterrows():
        link = str(r.get("link", "")).strip()
        if not link:
            continue
        code = str(r.get("course_code", "")).strip().upper()
        
        all_papers.append({
            "link": link,
            "title": tc(str(r.get("title", "")).strip()),
            "year": str(r.get("year", "")).strip(),
            "course_code": code,
        })
    
    with_code = [p for p in all_papers if p["course_code"]]
    print(f"Loaded {len(all_papers)} papers ({len(with_code)} with codes)")
    return all_papers, with_code

def main(dry_run=False):
    print("INGEST PAPERS")
    all_p, with_c = load_data()
    if dry_run:
        print(f"[DRY RUN] Would ingest {len(all_p)} papers")
        return

    driver = get_driver()
    try:
        with driver.session() as session:
            run_batch(session, PAPER_MERGE_QUERY, all_p, batch_size=500, label="papers")
            run_batch(session, LINK_QUERY, with_c, batch_size=500, label="links")
        print("\nDone!")
    finally:
        close_driver(driver)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    main(dry_run=p.parse_args().dry_run)
