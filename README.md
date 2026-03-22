# Chuka University GraphRAG Assistant 

An intelligent, context-aware academic ecosystem designed to transform how 15,000+ Chuka University students interact with institutional data.

---

## Key Features

- **Personalized AI Assistant**: A hybrid RAG system combining Graph (Neo4j) and Vector (FAISS) retrieval for pinpoint accuracy.
- **ChatGPT-Style Sidebar**: Persistent chat history with multi-session tracking, granular deletion, and seamless conversation restoration.
- **Multi-Modal Interaction**: Support for **Voice Dictation** (STT) and **PDF Document Analysis** (e.g., fee statements, personal timetables).
- **Matrix Timetable Generation**: Dynamic, grid-based personal schedules that map CourseUnits to real-time physical slots using Neo4j relationships.
- **Smart Fee Calculation**: Intelligent parsing of program fee structures to provide real-time cost breakdowns for students.
- **Exam Repository Access**: Direct, metadata-sorted links to past exam papers hosted on the official university repository.

---

##  How It Works: The Hybrid Architecture

The assistant leverages a **Dual-Layer Retrieval strategy** coordinated by the **Google Gemini 2.5 Flash** LLM:

1. **Structured Layer (Neo4j)**:
   - Models the university hierarchy: `Faculty ➔ Department ➔ Programme ➔ CourseUnit`.
   - Links units to `TimetableSlot` and `Room` nodes for real-time scheduling.
2. **Unstructured Layer (FAISS)**:
   - Vector index of the **Student Handbook**, **University Statutes**, and **Academic Policies**.
   - Uses the `all-mpnet-base-v2` embedding model for high-fidelity semantic search.
3. **Synthesis Engine**:
   - Classifies user intent and routes queries to the appropriate layer.
   - Grounded responses with zero hallucinations via strict context injection.

---

##  Project Structure

- `app.py`: High-fidelity Streamlit interface with dark-mode sidebar and integrated chat input.
- `src/chuka_graphrag_pipeline.py`: The core LLM orchestration engine with `tenacity` retry logic and modular Cypher handlers.
- `src/ingest_timetable.py`: Unified dual-mode ingestion tool for mapping CSV schedules to the Graph.
- `src/database.py`: Persistent user state management via PostgreSQL with SQLAlchemy connection pooling.
- `src/pdf_handler.py`: Specialized parser for Chuka-formatted academic documents.

---

##  Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Database Setup
- Ensure a **Neo4j AuraDB** (or local) instance is running.
- For production-style persistence, configure a **PostgreSQL** instance for chat history and profiles.
- If `DATABASE_URL` is not set, the app now falls back to a local SQLite database so Streamlit can still boot for demos or quick tests.
- Configure your `.env` file with the following keys:
  ```env
  GEMINI_API_KEY=your_gemini_key
  NEO4J_URI=bolt://...
  NEO4J_USERNAME=neo4j
  NEO4J_PASSWORD=your_password
  DATABASE_URL=postgresql://...
  ```

For Streamlit Cloud, add these under your app secrets:

- `GEMINI_API_KEY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `DATABASE_URL` if you want persistent multi-user chat history across restarts

### 3. Data Ingestion
Map the latest university timetable to the Knowledge Graph:
```bash
# Full wipe and re-ingest (for fresh semesters)
python src/ingest_timetable.py --mode full

# Incremental gap-fill (only map missing units)
python src/ingest_timetable.py --mode incremental
```

### 4. Run the Application
```bash
streamlit run app.py
```

---

##  Tech Stack
- **LLM**: Google Gemini 2.5 Flash
- **Knowledge Graph**: Neo4j (Cypher)
- **Vector DB**: FAISS
- **Backend**: Python 3.10, SQLAlchemy, PostgreSQL
- **Frontend**: Streamlit (Premium Custom CSS)
- **Document Processing**: PyPDF2, pdfplumber

---

## Authors
**Dancun Mau Wainaina**
