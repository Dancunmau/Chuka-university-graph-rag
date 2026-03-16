// Chuka University Knowledge Graph Schema
// Use these queries to initialize or verify your Neo4j instance

// 1. Constraints
CREATE CONSTRAINT IF NOT EXISTS FOR (p:Programme) REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:CourseUnit) REQUIRE c.code IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (f:Faculty) REQUIRE f.name IS UNIQUE;

// 2. Main Nodes
// (Faculty) -[:FACULTY_HAS_DEPARTMENT]-> (Department)
// (Department) -[:DEPARTMENT_OFFERS_PROGRAM]-> (Programme)
// (Programme) -[:HAS_UNIT {year, semester}]-> (CourseUnit)
// (CourseUnit) -[:HAS_PAST_PAPER]-> (PastPaper)
// (CourseUnit) -[:HAS_TIMESLOT]-> (TimetableSlot)
// (CourseUnit) -[:HAS_RESOURCE]-> (RepositoryItem)

// 3. Properties
// Programme: name, level, duration_string, fee_string
// CourseUnit: code, name, is_current
// TimetableSlot: day, time, room, day_index, start_time
// PastPaper: title, year, link, repository_id
