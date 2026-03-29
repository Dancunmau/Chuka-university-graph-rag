import re
import sys
import pandas as pd
from neo4j_utils import get_driver, get_project_root, run_batch, close_driver, tc

ROOT = get_project_root()
CSV_PATH = ROOT / "data" / "communities.csv"

def extract_course_codes(title):
    matches = re.findall(r'[A-Z]{2,5}\s*\d{3,4}', str(title).upper())
    return list({re.sub(r'([A-Z]+)\s*(\d+)', r'\1 \2', c) for c in matches})

ITEM_MERGE_QUERY = """
UNWIND $data AS row
MERGE (r:RepositoryItem {link: row.link})
SET r.title       = row.title,
    r.author      = row.author,
    r.type        = row.type,
    r.year        = row.year,
    r.community   = row.community,
    r.is_academic = row.is_academic
"""

LINK_QUERY = """
UNWIND $data AS row
MATCH (r:RepositoryItem {link: row.link})
MERGE (u:CourseUnit {code: row.code})
MERGE (u)-[:HAS_RESOURCE]->(r)
"""

def load_data():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH).fillna('')
    df.columns = [c.strip().lower() for c in df.columns]

    items_all, items_with_code = [], []
    for _, row in df.iterrows():
        link = str(row.get('repository_link', '')).strip()
        if not link:
            continue
            
        codes = extract_course_codes(row['title'])
        entry = {
            'link':      link,
            'title':     tc(str(row.get('title', '')).strip()),
            'author':    tc(str(row.get('author', '')).strip()),
            'type':      str(row.get('type', '')).strip(),
            'year':      str(row.get('year', '')).strip(),
            'community': str(row.get('community', '')).strip(),
            'is_academic': bool(row.get('community', '') in ['Conferences', 'Journals/Journal Articles', 'Research Articles']),
        }
        items_all.append(entry)
        for code in codes:
            items_with_code.append({'link': link, 'code': code})

    return items_all, items_with_code

def main(dry_run=False):
    print("INGEST COMMUNITIES")
    all_i, with_c = load_data()
    if dry_run:
        print(f"[DRY RUN] Would ingest {len(all_i)} items")
        return

    driver = get_driver()
    try:
        with driver.session() as s:
            run_batch(s, ITEM_MERGE_QUERY, all_i, batch_size=300, label="items")
            run_batch(s, LINK_QUERY, with_c, batch_size=300, label="links")
        print("\nDone!")
    finally:
        close_driver(driver)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    main(dry_run=p.parse_args().dry_run)
