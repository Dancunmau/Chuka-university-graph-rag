import os
import re
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# 1. Configuration
load_dotenv('d:/Jupyter notebook/Graph rag/.env')
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PWD  = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USER, PWD))

def parse_time_range(time_str):
    """Parses '7:00AM-10:00AM' into ('07:00', '10:00')"""
    try:
        pattern = r"(\d{1,2}:\d{2})\s*(AM|PM)"
        matches = re.findall(pattern, time_str, re.IGNORECASE)
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

def get_day_index(day):
    days = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
    return days.get(day.capitalize(), 0)

def ingest():
    csv_path = 'd:/Jupyter notebook/Graph rag/timetable.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Reading timetable.csv...")
    df = pd.read_csv(csv_path).fillna("Unknown")
    
    # Pre-process for efficiency
    processed = []
    for _, row in df.iterrows():
        start, end = parse_time_range(str(row['time']))
        processed.append({
            'day': str(row['day']).strip(),
            'day_index': get_day_index(str(row['day']).strip()),
            'room': str(row['room']).strip(),
            'time': str(row['time']).strip(),
            'start_time': start,
            'end_time': end,
            'unit_code': str(row['unit_code']).strip(),
            'semester': str(row.get('semester', '1')).strip()
        })

    with driver.session() as session:
        print("Wiping existing TimetableSlots...")
        session.run("MATCH (t:TimetableSlot) DETACH DELETE t")
        
        print(f"Ingesting {len(processed)} records (Dual-Layer Strategy)...")
        batch_size = 200
        total_ingested = 0
        program_linked = 0
        
        for i in range(0, len(processed), batch_size):
            batch = processed[i:i+batch_size]
            # Dual-Layer Cypher:
            # 1. Physical Layer: Always create Slot and link to Room.
            # 2. Academic Layer: Link Slot to Unit, and Unit to Programme (if exists).
            result = session.run("""
            UNWIND $data AS row
            
            // --- Physical Layer (Room Schedule) ---
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
            
            // --- Academic Layer (Curriculum) ---
            MERGE (u:CourseUnit {code: row.unit_code})
            MERGE (u)-[:HAS_TIMESLOT]->(t)
            
            // Optional: Count how many are actually linked to a programme 
            // (We don't create Programme relationships here, they should exist from curriculum ingestion)
            WITH u, t
            OPTIONAL MATCH (p:Programme)-[:HAS_UNIT]->(u)
            RETURN count(t) as total, count(p) as linked
            """, data=batch)
            
            stats = result.single()
            total_ingested += stats["total"]
            program_linked += stats["linked"]
            
        print(f"Ingestion complete.")
        print(f"- Total Slots: {total_ingested}")
        print(f"- Rooms created/merged: {session.run('MATCH (r:Room) RETURN count(r)').single()[0]}")
        
    driver.close()

if __name__ == "__main__":
    ingest()
