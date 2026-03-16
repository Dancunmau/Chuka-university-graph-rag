# Chuka University GraphRAG Assistant

An intelligent, context-aware academic ecosystem designed to transform how 15,000+ Chuka University students interact with institutional data.

---

## Mission
In the current higher education landscape, students often face **Information Fragmentation**. Critical data—timetables, fee structures, past papers, and university policies—is scattered across disconnected PDF files, Word documents, and web repositories.

This project implements a **GraphRAG (Graph Retrieval-Augmented Generation)** assistant that unifies these sources into a single, conversational interface. It solves:
- **Fragmentation**: One portal for all academic metadata.
- **Lack of Personalization**: Responses are tailored to your Program, Year, and Semester.
- **Manual Search**: Replaces 20-minute manual PDF searches with 2-second AI responses.

## How It Works: The Hybrid Brain
The assistant uses a "Dual-Layer" retrieval strategy to ensure maximum accuracy:

### 1. The Knowledge Graph (Structured) -> **Neo4j**
Models the university as a living network (Faculties > Departments > Programs > Units).
- **Use Case**: "What units do I take?" or "Who teaches COSC 121?".
- **Benefit**: Captures complex hierarchical relationships that standard AI usually misses.

### 2. The Semantic Index (Unstructured) -> **FAISS**
Indexes thousands of text chunks from the **Student Handbook** and **University Policies**.
- **Use Case**: "How do I defer my studies?" or "What is the exam policy for missed marks?".
- **Benefit**: Finds relevant procedures even if the user doesn't use the exact technical terms.

### 3. The LLM Synthesis -> **Google Gemini 2.0**
The "Director" that coordinates the two layers.
- It **classifies** your intent.
- It **extracts** entities (unit codes, dates).
- It **generates** a human-like response grounded in real data (no hallucinations).

---

## Key Functionalities
- **Smart Timetable**: Generates personalized schedules filtered by the current day or week.
- **Fee Calculator**: Automatically calculates total program costs based on real-time tuition data.
- **Exam Repository**: Direct links to past papers on `repository.chuka.ac.ke` with sorted metadata.
- **Privacy-First**: Identity is managed locally via anonymous device tokens—no passwords required.

---

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Configure your credentials in a `.env` file (Neo4j AuraDB + Gemini API Keys).

### Execution
```bash
streamlit run app.py
```

---

##  Technical Stack
- **Core Engine**: Python 3.10
- **Knowledge Graph**: Neo4j (Cypher)
- **Vector Database**: FAISS (all-mpnet-base-v2)
- **LLM**: Google Gemini 2.0 Flash
- **Web UI**: Streamlit
- **Database**: PostgreSQL (User State Persistence)

---

## Authors
**Dancun Mau Wainaina**

