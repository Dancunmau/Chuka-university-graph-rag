"""
ingest_timetable.py
====================
Unified timetable ingestion script for the Chuka University GraphRAG system.

Two modes:
  --mode full         Wipe all TimetableSlots and re-ingest everything (default)
  --mode incremental  Only create slots for CourseUnits that lack one

Usage:
  python src/ingest_timetable.py                  # full wipe + re-ingest
  python src/ingest_timetable.py --mode incremental  # gap-fill only
"""
import os
import re
import sys
import csv
import argparse
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
URI  = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PWD  = os.getenv("NEO4J_PASSWORD")

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'timetable.csv')

DAY_INDEXES = {
    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
    'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_time_range(time_str):
    """Parses '7:00AM-10:00AM' into ('07:00', '10:00')."""
    try:
        matches = re.findall(r"(\d{1,2}:\d{2})\s*(AM|PM)", time_str, re.IGNORECASE)
        if len(matches) < 2:
            return None, None
        times = []
        for time_part, period in matches:
            hour, minute = map(int, time_part.split(':'))
            period = period.upper()
            if period == 'PM' and hour < 12: hour += 12
            elif period == 'AM' and hour == 12: hour = 0
            times.append(f"{hour:02d}:{minute:02d}")
        return times[0], times[1]
    except Exception:
        return None, None

def normalize_code(raw: str) -> str:
    """Strip section suffixes like (A), (B), -P then normalise whitespace."""
    code = re.sub(r'\(.*?\)', '', raw)
    code = re.sub(r'-P$', '', code)
    code = re.sub(r'\s+', ' ', code)
    return code.strip().upper()

# ── Load CSV ──────────────────────────────────────────────────────────────────
def load_csv():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        sys.exit(1)
    
    rows = []
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            raw_code = row['unit_code'].strip()
            if '/' in raw_code:
                continue  # skip combined slots
            start, end = parse_time_range(row['time'].strip())
            rows.append({
                'day':        row['day'].strip(),
                'day_index':  DAY_INDEXES.get(row['day'].strip(), 0),
                'room':       row['room'].strip(),
                'time':       row['time'].strip(),
                'start_time': start,
                'end_time':   end,
                'unit_code':  normalize_code(raw_code),
                'raw_code':   raw_code,
                'semester':   row.get('semester', '1').strip() if 'semester' in row else '1',
            })
    print(f"Loaded {len(rows)} timetable entries from CSV.")
    return rows

# ── Full Ingest (wipe + rebuild) ──────────────────────────────────────────────
def full_ingest(driver, rows):
    with driver.session() as session:
        print("Wiping existing TimetableSlots...")
        session.run("MATCH (t:TimetableSlot) DETACH DELETE t")

        batch_size = 200
        total_ingested = 0
        program_linked = 0

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            result = session.run("""
            UNWIND $data AS row
            MERGE (r:Room {name: row.room})
            MERGE (t:TimetableSlot {
                day: row.day,
                room: row.room,
                time: row.time
            })
            SET t.start_time = row.start_time,
                t.end_time = row.end_time,
                t.day_index = row.day_index,
                t.semester = row.semester
            MERGE (t)-[:IN_ROOM]->(r)
            MERGE (u:CourseUnit {code: row.unit_code})
            MERGE (u)-[:HAS_TIMESLOT]->(t)
            WITH u, t
            OPTIONAL MATCH (p:Programme)-[:HAS_UNIT]->(u)
            RETURN count(t) as total, count(p) as linked
            """, data=batch)
            stats = result.single()
            total_ingested += stats["total"]
            program_linked += stats["linked"]

        rooms = session.run('MATCH (r:Room) RETURN count(r)').single()[0]
        print(f"Full ingestion complete.")
        print(f"  Total Slots created: {total_ingested}")
        print(f"  Programme-linked units: {program_linked}")
        print(f"  Rooms created/merged: {rooms}")

# ── Incremental Ingest (gap-fill only) ────────────────────────────────────────
def incremental_ingest(driver, rows):
    with driver.session() as session:
        unmapped = session.run("""
            MATCH (p:Programme)-[:HAS_UNIT]->(u:CourseUnit)
            WHERE NOT (u)-[:HAS_TIMESLOT]->(:TimetableSlot)
            RETURN DISTINCT u.code AS code
        """).data()

    unmapped_codes = {r['code'].strip().upper() for r in unmapped}
    print(f"CourseUnits in Programmes without timeslots: {len(unmapped_codes)}")

    to_create = [r for r in rows if r['unit_code'] in unmapped_codes]
    print(f"CSV rows matched to unmapped units: {len(to_create)}")

    created = 0
    with driver.session() as session:
        for row in to_create:
            session.run("""
                MATCH (u:CourseUnit {code: $code})
                MERGE (r:Room {name: $room})
                MERGE (t:TimetableSlot {
                    day: $day, room: $room, time: $time
                })
                SET t.start_time = $start_time,
                    t.end_time = $end_time,
                    t.day_index = $day_index,
                    t.semester = $semester
                MERGE (t)-[:IN_ROOM]->(r)
                MERGE (u)-[:HAS_TIMESLOT]->(t)
            """, code=row['unit_code'], day=row['day'], room=row['room'],
                time=row['time'], start_time=row['start_time'],
                end_time=row['end_time'], day_index=row['day_index'],
                semester=row['semester'])
            created += 1

    # Summary
    with driver.session() as session:
        still = session.run("""
            MATCH (p:Programme)-[:HAS_UNIT]->(u:CourseUnit)
            WHERE NOT (u)-[:HAS_TIMESLOT]->(:TimetableSlot)
            RETURN count(DISTINCT u) AS ct
        """).data()[0]['ct']
    print(f"\nIncremental ingestion complete!")
    print(f"  Timeslot records created: {created}")
    print(f"  Units still without timeslot: {still}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Timetable CSV → Neo4j Ingestion")
    parser.add_argument('--mode', choices=['full', 'incremental'], default='full',
                        help="'full' wipes and rebuilds all slots. 'incremental' only fills gaps.")
    args = parser.parse_args()

    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    rows = load_csv()

    if args.mode == 'full':
        full_ingest(driver, rows)
    else:
        incremental_ingest(driver, rows)

    driver.close()
    print("Done.")

if __name__ == "__main__":
    main()
