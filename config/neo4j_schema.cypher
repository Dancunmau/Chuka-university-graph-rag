// Chuka University Knowledge Graph — Schema


// Constraints (Uniqueness)

CREATE CONSTRAINT IF NOT EXISTS FOR (f:Faculty)        REQUIRE f.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Department)     REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Program)        REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (u:CourseUnit)     REQUIRE u.code IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (p:PastPaper)      REQUIRE p.link IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (r:Room)           REQUIRE r.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (r:RepositoryItem) REQUIRE r.link IS UNIQUE;

// Composite constraint for TimetableSlot (Day, Room, Time, Semester must be unique)
CREATE CONSTRAINT IF NOT EXISTS FOR (t:TimetableSlot) 
REQUIRE (t.day, t.room, t.time, t.semester) IS UNIQUE;

// Indices (Performance)

CREATE INDEX IF NOT EXISTS FOR (u:CourseUnit) ON (u.name);
CREATE INDEX IF NOT EXISTS FOR (p:Program)    ON (p.level);

// Node Reference
//
// Faculty:        name                             (Title Case, e.g. "Faculty of Science & Technology")
// Department:     name                             (Title Case, e.g. "Department of Computer Sciences")
// Program:        name, level, duration_string, fee_string, units_ingested (bool)
// CourseUnit:     code, name, is_current (bool)
// PastPaper:      link, title, year, course_code
// TimetableSlot:  day, time, semester, room, day_index, start_time, end_time
// Room:           name
// RepositoryItem: link, title, author, type, year, community

// Relationship Reference
//
// (Faculty)      -[:HAS_DEPARTMENT]->   (Department)
// (Department)   -[:OFFERS_PROGRAM]->   (Program)
// (Program)      -[:HAS_UNIT {year, semester}]-> (CourseUnit)
// (CourseUnit)   -[:HAS_PAST_PAPER]->   (PastPaper)
// (CourseUnit)   -[:HAS_TIMESLOT]->     (TimetableSlot)
// (TimetableSlot)-[:IN_ROOM]->          (Room)
// (CourseUnit)   -[:HAS_RESOURCE]->     (RepositoryItem)
