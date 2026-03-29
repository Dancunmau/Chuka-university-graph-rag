import re
import sys
import csv
import argparse
from neo4j_utils import get_driver, get_project_root, close_driver, tc

CSV_PATH = get_project_root() / "data" / "timetable.csv"

DAY_INDEX = {
    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
    'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7,
}

def parse_time_range(time_str: str):
    try:
        matches = re.findall(r'(\d{1,2}:\d{2})\s*(AM|PM)', time_str, re.IGNORECASE)
        if len(matches) < 2:
            return None, None
        times = []
        for time_part, period in matches:
            h, m = map(int, time_part.split(':'))
            period = period.upper()
            if period == 'PM' and h < 12:
                h += 12
            elif period == 'AM' and h == 12:
                h = 0
            times.append(f"{h:02d}:{m:02d}")
        return times[0], times[1]
    except Exception:
        return None, None

def normalise_code(raw: str) -> str:
    code = re.sub(r'\(.*?\)', '', raw)
    code = re.sub(r'-P$', '', code)
    return re.sub(r'\s+', ' ', code).strip().upper()

def load_data():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        sys.exit(1)

    rows = []
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            raw_code = row['unit_code'].strip()
            if '/' in raw_code:
                continue
            start, end = parse_time_range(row['time'].strip())
            rows.append({
                'day': row['day'].strip(),
                'day_index': DAY_INDEX.get(row['day'].strip(), 0),
                'room': row['room'].strip().upper(),
                'time': row['time'].strip(),
                'start_time': start,
                'end_time': end,
                'unit_code': normalise_code(raw_code),
                'semester': row.get('semester', '1').strip(),
            })
    print(f"Loaded {len(rows)} timetable entries")
    return rows

MERGE_QUERY = """
UNWIND $data AS row
MERGE (u:CourseUnit {code: row.unit_code})
MERGE (r:Room {name: row.room})
MERGE (t:TimetableSlot {day: row.day, room: row.room, time: row.time, semester: row.semester})
SET t.start_time = row.start_time,
    t.end_time   = row.end_time,
    t.day_index  = row.day_index
MERGE (t)-[:IN_ROOM]->(r)
MERGE (u)-[:HAS_TIMESLOT]->(t)
"""

def main(dry_run=False):
    print("INGEST TIMETABLE")
    rows = load_data()
    if dry_run:
        print(f"[DRY RUN] Would ingest {len(rows)} timetable rows")
        return

    driver = get_driver()
    try:
        print("Wiping existing TimetableSlots...")
        with driver.session() as s:
            s.run("MATCH (t:TimetableSlot) DETACH DELETE t")
        with driver.session() as s:
            s.run(MERGE_QUERY, data=rows)
        print("\nDone!")
    finally:
        close_driver(driver)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    main(dry_run=p.parse_args().dry_run)
