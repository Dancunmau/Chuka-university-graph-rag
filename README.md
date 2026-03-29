# Chuka University GraphRAG Assistant

**Empowering 15,000+ students with an intelligent, context-aware academic ecosystem.**

The Chuka University Assistant is a state-of-the-art **Hybrid RAG (Retrieval-Augmented Generation)** platform. It bridges the gap between structured institutional data (Neo4j Knowledge Graph) and unstructured academic policies (FAISS Vector Store) to provide students with a single, accurate point of truth for their university experience.

---

##  Vision: The Student Command Center

Our goal is to eliminate the "Information Desert" by giving every student a personalized AI counselor that understands their program, their year of study, and their daily schedule.

### **Key Strategic Features**
-  **Hybrid Intelligence**: Combines **Neo4j Cypher** logic for structured data (who teaches what, when, and where) with **FAISS vector search** for policy documents (Student Handbook, Statutes).
-  **Time-Aware Conversationalist**: The assistant is contextually aware of "now." It translates queries like *"What is my schedule today?"* or *"Where is my class tomorrow?"* into precise database lookups.
-  **Voice & Multi-Modal**: Students can dictate queries via **Speech-to-Text** or upload PDF documents (fee statements, personal timetables) for instant AI analysis.
-  **Personalized Sidebar**: A "ChatGPT-style" interface that persists session history, generates on-the-fly PDF timetables, and manages academic identities.
-  **Examination & Community Hub**: Direct integration with the university repository, providing instant links to past papers and community-curated study resources.

---

##  The Hybrid Engine: How it Works

The platform operates on a **Dual-Layer Retrieval** strategy orchestrated by **Google Gemini 2.5 Flash**:

1. **The Knowledge Graph (Neo4j)**:
   - Maps the university from the top down: `Faculty ➔ Department ➔ Programme ➔ CourseUnit`.
   - Connects units to `TimetableSlot` and `Room` nodes for real-time schedule resolution.
   - Links `CourseUnit` to `PastPaper` and `RepositoryItem` for resource discovery.

2. **The Vector Store (FAISS)**:
   - Indexes unstructured documents using the `all-mpnet-base-v2` embedding model.
   - Allows students to query the **Student Handbook** as if they were talking to an expert.

3. **Autonomous Synthesis**:
   - The system classifies student intent and extracts academic entities (course codes, semesters, years) with regex-enhanced LLM logic.
   - Responses are **strictly grounded**; if the data isn't in the graph or the vector store, the AI will not hallucinate.

---

##  Project Structure

- `app.py`: The high-fidelity Streamlit interface with premium custom CSS and side-panel persistent history.
- `src/chuka_graphrag_pipeline.py`: The "Heart" of the system. Manages intent classification, entity extraction, and multi-source retrieval.
- `src/ingest_all.py`: The Master Orchestrator. A dependency-safe tool that populates the entire graph in 5 automated steps.
- `src/database.py`: PostgreSQL backend for managing multi-session chat history and user state.
- `src/pdf_handler.py`: Specialized Chuka-native document parser for student-uploaded PDFs.

---

##  Getting Started

### 1. Installation & Environment
```bash
# Clone the repository and install dependencies
pip install -r requirements.txt
```

Configure your `.env` with your API keys and credentials:
```env
GEMINI_API_KEY=your_key
NEO4J_URI=bolt://...
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
DATABASE_URL=postgresql://...
```

### 2. The Ingestion Workflow (Master Orchestrator)
Populate your Knowledge Graph with the specialized university dataset:
```bash
# Full automated ingestion with database wipe
python src/ingest_all.py --wipe

# Or run specific steps (e.g., only update past papers)
python src/ingest_all.py --steps 3
```

### 3. Build the Vector Index
```bash
# Create the semantic index for handbooks and policies
python src/build_faiss_index.py
```

### 4. Launch the Assistant
```bash
streamlit run app.py
```

---

##  Tech Stack
-   **LLM**: Google Gemini 2.5 Flash (Synthesizer & Intent Engine)
-   **Knowledge Graph**: Neo4j Aura (Relationship Modeling)
-   **Vector Search**: FAISS (SentenceTransformer Embeddings)
-   **Persistence**: PostgreSQL (SQLAlchemy)
-   **Interface**: Streamlit 
-   **PDF Engine**: ReportLab (Generation), PDFPlumber (Extraction)

---

##  Data Coverage Notice
> [!IMPORTANT]
> Currently, the **Timetable Coverage** stands at approximately **22%** for unique units in the curriculum. While the system understands all 900+ course units, specific "Time and Room" data is limited by the current source `timetable.csv`. The assistant handles these data gaps gracefully by providing unit descriptions when schedules are unavailable.

---

## Author
**Dancun Mau Wainaina**
