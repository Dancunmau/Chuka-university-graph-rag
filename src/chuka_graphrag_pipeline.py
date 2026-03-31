
"""
Orchestrates the GraphRAG pipeline for the Chuka University assistant.

Architecture flow:
  1. classify_intent(): Determines whether to query the graph, vector store, or both.
  2. extract_entities(): Extracts structured parameters (course, year, etc.) using Gemini with regex fallback.
  3. retrieve_from_graph(): Executes targeted Cypher queries in Neo4j.
  4. retrieve_from_faiss(): Performs semantic search across policy documents (if available).
  5. synthesise_response(): Grounds the final LLM response using retrieved context.
  6. GraphRAGAssistant.generate_response(): Main pipeline entry point.
"""


import os
import re
import json
import pickle
import logging
import numpy as np
import datetime
import typing_extensions as typing
from tenacity import retry, wait_exponential, stop_after_attempt
import threading

from dotenv import load_dotenv
import google.generativeai as genai
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ChukaPipeline")

# FAISS dependency
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    log.warning("FAISS/Sentence-Transformers missing — semantic search disabled.")

# API Configuration
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

NEO4J_URI      = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY", ""),
    os.getenv("GEMINI_API_KEY2", ""),
    os.getenv("GEMINI_API_KEY3", "")
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k] # Filter out empty keys
current_key_idx = 0
FAISS_INDEX_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index.bin")
FAISS_METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_metadata.pkl")


# Thread lock for thread-safe API key rotation
key_lock = threading.Lock()

# fallback models
FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"]

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
def _gemini_call(prompt_or_contents, model_name=None, stream=False):
    """
    Call Gemini with back-off via tenacity,
    cascading model fallbacks, and thread-safe key rotation.
    Returns string if stream=False, or the response object if stream=True.
    """
    global current_key_idx
    
    # Selection of models to try
    models_to_try = [model_name] if model_name else FALLBACK_MODELS
    
    for m_name in models_to_try:
        try:
            with key_lock:
                key_to_use = GEMINI_KEYS[current_key_idx]
                
            genai.configure(api_key=key_to_use)
            model = genai.GenerativeModel(m_name)
            response = model.generate_content(prompt_or_contents, stream=stream)
            if stream:
                return response
            return response.text.strip()
            
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                log.warning(f"Quota exceeded on current key for model {m_name}. Rotating key...")
                with key_lock:
                    if len(GEMINI_KEYS) > 1:
                        current_key_idx = (current_key_idx + 1) % len(GEMINI_KEYS)
                #trigger Tenacity retry so it starts over with the primary model on the new key
                raise Exception("Quota exceeded, key rotated. Retrying from top")
            elif "503" in err:
                log.warning(f"Model {m_name} failed. Attempting next model")
                continue # Try next model name in the fallback cascade
            else:
                log.error(f"Gemini Model Error ({m_name}): {err}")
                raise e

    raise Exception("All tiered models exhausted")


# Retrieval Routing
def classify_intent(query: str) -> str:
    """
    Returns one of: 'graph_query' | 'semantic_search' | 'hybrid'
    - graph_query : questions about units, timetable slots, past papers by code
    - semantic_search : policy, handbook, fee, procedure questions
    - hybrid : mixed (e.g. 'units and their exam regulations')
    """
    prompt = f"""Classify the following student query to determine the best data retrieval strategy for the Chuka University assistant.

Query: "{query}"

Classification Rules:
- "graph_query": The user is asking about highly structured academic data such as specific course units, timetable schedules, rooms, or past paper availability.
- "semantic_search": The user is asking about general rules, handbook policies, fee structures, graduation procedures, or university regulations.
- "hybrid": The user's query requires both structured database lookup AND general handbook context (e.g., "What are the exam rules for COSC 121?").

You must respond with EXACTLY ONE of the following three core labels. Do not include any other text, punctuation, or explanations. 

Labels: graph_query, semantic_search, hybrid"""

    result = _gemini_call(prompt).strip().lower()
    for label in ["graph_query", "semantic_search", "hybrid"]:
        if label in result:
            return label
    # Default: try graph first
    return "hybrid"


def extract_entities(query: str, profile: dict) -> dict:
    """
    Extract structured entities from the query.
    Returns dict with keys: course_code, programme, year, semester
    Falls back to regex parse if the LLM output deviates from JSON.
    """
    now = datetime.datetime.now()
    current_day = now.strftime("%A")
    tomorrow = (now + datetime.timedelta(days=1)).strftime("%A")

    prompt = f"""Extract academic entities from this Chuka University student query.
Query: "{query}"
Student profile: {json.dumps(profile or {{}})}

Current Date Context:
- Today is: {current_day}
- Tomorrow is: {tomorrow}
(Translate any relative terms like "today" or "tomorrow" in the query into their actual day of the week)

Return valid JSON only, with these keys (use null if not found):
{{
  "course_code": "e.g. COSC 121 or null",
  "programme": "e.g. Computer Science or null",
  "year": "integer 1-6 or null",
  "semester": "integer 1-2 or null",
  "day": "e.g. Monday or null",
  "topic": "general topic if no specific code, e.g. exam regulations"
}}"""

    raw = _gemini_call(prompt)
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        entities = json.loads(raw)
    except Exception as e:
        log.error(f"Structured extraction failed: {e}. Using regex fallback.")
        entities = {"course_code": None, "programme": None, "year": None, "semester": None, "day": None, "topic": None}
        m_code = re.search(r'\b([A-Z]{3,5})\s*(\d{3,4})\b', query.upper())
        if m_code:
            entities["course_code"] = f"{m_code.group(1)} {m_code.group(2)}".upper()
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for d in days:
            if d.lower() in query.lower():
                entities["day"] = d
                break

    # Fill from profile where missing
    if profile:
        if not entities.get("year") and profile.get("year"):
            entities["year"] = profile["year"]
        if not entities.get("semester") and profile.get("semester"):
            entities["semester"] = profile["semester"]
        if not entities.get("programme") and profile.get("program"):
            entities["programme"] = profile["program"]

    return entities



# Neo4j Cypher Helpers
def _query_past_papers(session, query: str, entities: dict) -> list:
    results = []
    code = entities.get("course_code")
    topic = entities.get("topic")
    if code:
        r = session.run("""
        MATCH (c:CourseUnit)-[:HAS_PAST_PAPER]->(p:PastPaper)
        WHERE c.code =~ $pattern
        RETURN c.code as code, c.name as name,
               p.title as title, p.year as year, p.link as link
        ORDER BY p.year DESC LIMIT 10
        """, pattern=f"(?i).*{re.escape(code)}.*")
        rows = r.data()
        if rows:
            results.append(f"Past papers for {code}:")
            for row in rows:
                results.append(f"  - [{row['year']}] {row['title']}\n    Link: {row['link']}")
    elif topic and "past paper" in str(query).lower():
        r = session.run("""
        MATCH (c:CourseUnit)-[:HAS_PAST_PAPER]->(p:PastPaper)
        WHERE p.title =~ $pattern
        RETURN c.code as code, c.name as name,
               p.title as title, p.year as year, p.link as link
        ORDER BY p.year DESC LIMIT 10
        """, pattern=f"(?i).*{re.escape(topic)}.*")
        rows = r.data()
        if rows:
            results.append(f"Past papers matching '{topic}':")
            for row in rows:
                results.append(f"  - [{row['year']}] {row['title']}\n    Link: {row['link']}")
    return results

def _query_units(session, entities: dict) -> list:
    results = []
    prog = entities.get("programme")
    year = entities.get("year")
    sem = entities.get("semester")
    day = entities.get("day")
    
    if not prog: return results
    
    params = {"prog": prog}
    cypher = """
    MATCH (p:Program)-[r:HAS_UNIT]->(u:CourseUnit)
    WHERE toLower(p.name) CONTAINS toLower($prog)
    """
    if year:
        cypher += " AND r.year = $year"
        params["year"] = f"Year {year}"
    if sem:
        cypher += " AND r.semester = $sem"
        params["sem"] = str(sem)

    cypher += "\nOPTIONAL MATCH (u)-[:HAS_TIMESLOT]->(t:TimetableSlot)"
    if day:
        cypher += " WHERE toLower(t.day) = toLower($day)"
        params["day"] = day

    cypher += """
    RETURN u.code as code, u.name as name,
           r.year as level, r.semester as semester,
           collect(t.day + " " + t.time + " (" + t.room + ")") as timeslots
    ORDER BY u.code LIMIT 20
    """
    r = session.run(cypher, **params)
    rows = r.data()
    if rows:
        header = f"Units for {prog}"
        if year: header += f" Year {year}"
        if sem:  header += f" Sem {sem}"
        results.append(header + ":")
        for row in rows:
            slot_str = ""
            if row.get('timeslots') and any(row['timeslots']):
                slot_str = " | ".join([s for s in row['timeslots'] if s and "None" not in s])
                slot_str = f"\n    Timetable: {slot_str}"
            results.append(f"  - {row['code']}: {row['name']}{slot_str}")
    return results

def _query_fees(session, query: str, entities: dict) -> list:
    results = []
    prog = entities.get("programme")
    if not prog: return results
    
    fee_keywords = ["fee", "cost", "pay", "tuition", "amount", "price", "charge", "how much"]
    wants_fees = any(kw in str(query).lower() for kw in fee_keywords)
    if wants_fees:
        r = session.run("""
        MATCH (p:Program)
        WHERE toLower(p.name) CONTAINS toLower($prog)
        RETURN p.name as name, p.fee_string as fee, p.duration_string as duration
        """, prog=prog)
        rows = r.data()
        if rows:
            p_data = rows[0]
            fee_info = p_data.get('fee') or 'Not listed'
            dur_info = p_data.get('duration') or 'Not listed'
            results.append(f"Fee Structure for {p_data['name']}:")
            results.append(f"  - Fees: {fee_info}")
            results.append(f"  - Duration: {dur_info}")
            
            # FR7.2: Total Cost Calculation
            if "total" in str(query).lower() or "how much" in str(query).lower():
                try:
                    amount = re.search(r'(\d{1,3}(?:,\d{3})*)', fee_info.replace(',', ''))
                    num_sems = re.search(r'(\d+)', dur_info)
                    if amount and num_sems:
                        total = int(amount.group(1).replace(',', '')) * int(num_sems.group(1))
                        results.append(f"  - Estimated Total Program Cost: KES {total:,}")
                except: pass
    return results

def _query_resources(session, query: str, entities: dict) -> list:
    results = []
    code = entities.get("course_code")
    topic = entities.get("topic")
    if code:
        r = session.run("""
        MATCH (c:CourseUnit)-[:HAS_RESOURCE]->(ri:RepositoryItem)
        WHERE toLower(c.code) CONTAINS toLower($code)
        RETURN ri.title as title, ri.author as author,
               ri.type as type, ri.year as year,
               ri.community as community, ri.link as link
        ORDER BY ri.year DESC LIMIT 8
        """, code=code)
        rows = r.data()
        if rows:
            results.append(f"Study resources for {code}:")
            for row in rows:
                author_str = f" — {row['author']}" if row.get('author') else ""
                results.append(
                    f"  - [{row.get('type','Resource')}] {row['title']}{author_str}\n"
                    f"    Link: {row['link']}"
                )
    elif topic:
        r = session.run("""
        MATCH (ri:RepositoryItem)
        WHERE ri.title =~ $pattern
        RETURN ri.title as title, ri.author as author,
               ri.type as type, ri.year as year,
               ri.community as community, ri.link as link
        ORDER BY ri.year DESC LIMIT 8
        """, pattern=f"(?i).*{re.escape(topic)}.*")
        rows = r.data()
        if rows:
            results.append(f"Repository resources matching '{topic}':")
            for row in rows:
                author_str = f" — {row.get('author','')}" if row.get('author') else ""
                results.append(
                    f"  - [{row.get('type','Resource')}] {row['title']}{author_str}\n"
                    f"    Link: {row['link']}"
                )
    return results

def _query_current_units(session) -> list:
    results = []
    r = session.run("""
    MATCH (u:CourseUnit)-[:HAS_TIMESLOT]->(t:TimetableSlot)
    WHERE u.is_current = True
    RETURN u.code as code, u.name as name, collect(t.day + " " + t.time + " (" + t.room + ")") as timeslots
    LIMIT 15
    """)
    rows = r.data()
    if rows:
        results.append("Units currently offered (Jan-April 2026):")
        for row in rows:
            slot_str = " | ".join([s for s in row['timeslots'] if s and "None" not in s])
            results.append(f"  - {row['code']}: {row['name']}\n    Timetable: {slot_str}")
    return results

def _query_catalogue(session, query: str) -> list:
    results = []
    faculty_keywords = {
        "science":     "Science",
        "technology":  "Science",
        "business":    "Business",
        "economics":   "Business",
        "education":   "Education",
        "agriculture": "Agriculture",
        "environment": "Agriculture",
        "humanities":  "Humanities",
        "social":      "Humanities",
        "nursing":     "Nursing",
        "health":      "Nursing",
        "law":         "Law",
        "engineering": "Engineering",
    }
    q_lower = query.lower()
    faculty_filter = next(
        (label for kw, label in faculty_keywords.items() if kw in q_lower),
        None
    )

    if faculty_filter:
        r = session.run("""
        MATCH (f:Faculty)-[:HAS_DEPARTMENT]->(d:Department)
              -[:OFFERS_PROGRAM]->(p:Program)
        WHERE f.name =~ $pattern
        RETURN f.name AS faculty, d.name AS department, p.name AS programme
        ORDER BY d.name, p.name
        """, pattern=f"(?i).*{faculty_filter}.*")
        header = f"Programmes offered (Faculty of {faculty_filter}):"
    else:
        r = session.run("""
        MATCH (f:Faculty)-[:HAS_DEPARTMENT]->(d:Department)
              -[:OFFERS_PROGRAM]->(p:Program)
        RETURN f.name AS faculty, d.name AS department, p.name AS programme
        ORDER BY f.name, d.name, p.name
        """)
        header = "All programmes offered at Chuka University:"

    rows = r.data()
    if rows:
        results.append(header)
        current_dept = None
        for row in rows:
            if row['department'] != current_dept:
                current_dept = row['department']
                results.append(f"\n  [{current_dept}]")
            results.append(f"    - {row['programme']}")
    return results

# Core Retrieval Orchestration

def retrieve_from_graph(query: str, entities: dict, profile: dict, driver) -> str:
    """Run Cypher queries against Neo4j by routing to specific handler functions."""
    results = []

    try:
        with driver.session() as session:
            # 0. Student Academic Identity Context
            prog = entities.get("programme") or profile.get("program") or profile.get("programme")
            if prog:
                r_identity = session.run("""
                MATCH (f:Faculty)-[:HAS_DEPARTMENT]->(d:Department)-[:OFFERS_PROGRAM]->(p:Program)
                WHERE p.name = $prog
                RETURN f.name AS faculty, d.name AS department
                """, prog=prog)
                id_row = r_identity.single()
                if id_row:
                    results.append(f"Student Academic Identity - Programme: {prog} | Department: {id_row['department']} | Faculty: {id_row['faculty']}")

            # (A) Past papers
            results.extend(_query_past_papers(session, query, entities))
            
            # (B) & (C) Units & Fees
            results.extend(_query_units(session, entities))
            results.extend(_query_fees(session, query, entities))
            
            # (D) Repository items
            resource_keywords = ["note", "material", "resource", "lecture", "paper", "book", "chapter", "article", "research", "find", "read", "document"]
            wants_resource = any(kw in str(query).lower() for kw in resource_keywords)
            if wants_resource or not results:
                results.extend(_query_resources(session, query, entities))
            
            # Current Units
            code = entities.get("course_code")
            prog = entities.get("programme")
            if (not code and not prog) or "current" in str(entities.get("topic","")).lower() or "this semester" in str(profile).lower():
                if len(results) < 3: 
                    results.extend(_query_current_units(session))
            
            # (E) Catalogue
            catalogue_keywords = ["what programme", "what program", "programmes offered", "programs offered", "what can i study", "what do you offer", "available programme", "available program", "list of programme", "list of program", "courses offered", "what courses", "which programme", "which program", "study at chuka", "offered at chuka"]
            wants_catalogue = any(kw in str(query).lower() for kw in catalogue_keywords)
            if wants_catalogue and not results:
                results.extend(_query_catalogue(session, query))

    except Exception as e:
        err_str = str(e)
        log.error(f"Graph retrieval error: {err_str}")
        if "Cannot resolve address" in err_str or "ServiceUnavailable" in err_str:
            log.warning("Neo4j database appears to be offline or unreachable. Check NEO4J_URI in .env")

    return "\n".join(results) if results else ""


def retrieve_from_faiss(query: str, index, metadata: list, embedder, k=8) -> str:
    """Search FAISS index for relevant handbook/policy chunks."""
    if not FAISS_AVAILABLE or index is None:
        return ""
    try:
        vec = embedder.encode([query]).astype(np.float32)
        faiss.normalize_L2(vec)
        _, ids = index.search(vec, k)
        chunks = []
        for idx in ids[0]:
            if 0 <= idx < len(metadata):
                m = metadata[idx]
                source = m.get('source', 'Unknown Document')
                page = m.get('page', '?')
                # The text already contains the header/page prefix from build_faiss_index.py
                chunks.append(f"--- Document: {source} (Page {page}) ---\n{m.get('text','')}")
        return "\n\n".join(chunks)
    except Exception as e:
        log.error(f"FAISS retrieval error: {e}")
        return ""



# LLM Synthesis
def synthesise_response(query: str, graph_ctx: str, faiss_ctx: str, profile: dict, extra_ctx: str = "", stream: bool = False):
    """Synthesise a grounded response using retrieved data."""
    profile_str = json.dumps(profile or {})
    context = ""
    if extra_ctx:
        context += f"=== Information from Your Uploaded Document ===\n{extra_ctx}\n\n"
    if graph_ctx:
        context += f"=== Academic Database Records (Neo4j) ===\n{graph_ctx}\n\n"
    if faiss_ctx:
        context += f"=== Handbook / Policy Context ===\n{faiss_ctx}\n\n"
        
    if not context:
        context = "No specific data was found in the database for this query."

    prompt = f"""You are the Chuka University Academic Assistant — a helpful, accurate chatbot for students.
Your goal is to provide precise answers based on the provided context.

User Profile: {profile_str}
Student Query: "{query}"

Context:
{context}

Instructions:
1. Use ONLY the provided context to answer. If the answer is not in the context, say you don't know but suggest who to contact (e.g., Registrar or specify the department).
2. PRIORITIZE information from the 'Uploaded Document' if it answers the student's specific question (e.g., personal timetable/fees).
3. When using information from the 'Handbook / Policy Context', ALWAYS cite the document name and page number.
3. If multiple versions of information exist, prioritize the most recent (e.g., 2026 Timetable over 2024 Advert).
4. Format your response using markdown. Use bullet points for lists.
5. Be professional yet friendly.
6. If past paper links are available, list them clearly with the year and direct URL.
7. If units are listed, format them as a numbered list.
8. IF YOU ARE ASKED ABOUT FEES: Use the provided fee structure and duration to calculate the TOTAL estimated cost of the programme if possible. Show your math clearly.
9. If no relevant data was found, honestly say so and suggest the student visits the registry or repository.chuka.ac.ke.
10. Be concise, warm, and student-friendly.
11. End with a follow-up question if appropriate.

Response:"""

    answer = _gemini_call(prompt, stream=stream)
    if answer:
        return answer

    log.error(f"Gemini synthesis failed for query: {query}. Falling back to friendly message.")

    # Build a clean callback
    if graph_ctx:
        # Graph found structured data 
        lines = [l.strip() for l in graph_ctx.split("\n") if l.strip() and not l.startswith("[")]
        excerpt = "\n".join(lines[:15])
        return (
            "The AI assistant is temporarily unavailable, but here is the information found in the database:\n\n"
            f"{excerpt}\n\n"
            "_For full details, please contact the Registrar or visit [repository.chuka.ac.ke](https://repository.chuka.ac.ke)._"
        )
    elif faiss_ctx:
        # FAISS found document chunks — show the first useful non-header line
        lines = [l.strip() for l in faiss_ctx.split("\n") if l.strip() and not l.startswith("[") and not l.startswith("---")]
        excerpt = " ".join(lines[:6])
        return (
            " The AI assistant is temporarily unavailable. Based on the document search, here is a relevant excerpt:\n\n"
            f"_{excerpt}_\n\n"
            "_Please contact the Registrar or visit the Student Handbook for complete information._"
        )

    return (
        "I'm sorry, I couldn't find specific information for your query right now. "
        "Please try rephrasing, or contact the relevant office (e.g., the Registrar or your Department) directly."
    )


# Main Assistant Class
class GraphRAGAssistant:
    """Public API used by app.py"""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")
        genai.configure(api_key=GEMINI_API_KEY)

        if not NEO4J_URI:
            raise ValueError("NEO4J_URI not set in .env")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

        # FAISS 
        self.faiss_index = None
        self.faiss_meta  = []
        self.embedder    = None
        if FAISS_AVAILABLE:
            self._load_faiss()

        log.info("GraphRAGAssistant initialised.")

    def _load_faiss(self):
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
            try:
                self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
                with open(FAISS_METADATA_PATH, "rb") as f:
                    self.faiss_meta = pickle.load(f)
                self.embedder = SentenceTransformer("all-mpnet-base-v2")
                log.info(f"FAISS loaded: {self.faiss_index.ntotal} vectors")
            except Exception as e:
                log.warning(f"FAISS load failed: {e}")
        else:
            log.warning("FAISS index not found — semantic search disabled. Run build_faiss_index.py.")

    def populate_vector_store(self, docs: list):
        """Add new documents to the FAISS index (called from app.py upload)."""
        if not FAISS_AVAILABLE:
            return
        if self.embedder is None:
            self.embedder = SentenceTransformer("all-mpnet-base-v2")

        texts = [d["text"] for d in docs]
        vecs  = self.embedder.encode(texts).astype(np.float32)
        faiss.normalize_L2(vecs)

        if self.faiss_index is None:
            dim = vecs.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)

        self.faiss_index.add(vecs)
        self.faiss_meta.extend(docs)

        faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, "wb") as f:
            pickle.dump(self.faiss_meta, f)
        log.info(f"FAISS updated: {len(docs)} new chunks added.")

    def generate_response(self, query: str, user_profile: dict, extra_context: str = "") -> str:
        """The main pipeline entry point."""
        gen = self.generate_response_stream(query, user_profile, extra_context)
        return "".join(list(gen))

    def generate_response_stream(self, query: str, user_profile: dict, extra_context: str = ""):
        """Generator that yields text chunks continuously as Gemini streams them."""
        student_name = user_profile.get("full_name", "Student")
        
        # 1. Intent Classification
        intent = classify_intent(query)
        log.info(f"Query Intent: {intent}")
        
        # 2. Entity Extraction
        # Pass both query and profile to match definition
        entities = extract_entities(query, user_profile) 
        log.info(f"Extracted Entities: {entities}")
        
        # 3. Graph Retrieval
        graph_nodes = retrieve_from_graph(query, entities, user_profile, self.driver)
        
        # 4. FAISS Retrieval
        faiss_results = ""
        if FAISS_AVAILABLE:
            faiss_results = retrieve_from_faiss(query, self.faiss_index, self.faiss_meta, self.embedder)
            
        # 5. Synthesis
        response_obj = synthesise_response(
            query, 
            graph_nodes, 
            faiss_results,
            user_profile,
            extra_context,
            stream=True
        )
        
        if isinstance(response_obj, str):
            yield response_obj
        else:
            for chunk in response_obj:
                try:
                    if chunk.text:
                        yield chunk.text
                except Exception as e:
                    pass

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes using Gemini 2.5 Flash."""
        audio_part = {
            "mime_type": "audio/wav", # Streamlit audio_input records as wav
            "data": audio_bytes
        }
        prompt = "Transcribe this audio clip of a student query. Return only the transcription text."
        # Rely on the _gemini_call for locks, retries, and key rotation
        return _gemini_call([prompt, audio_part], model_name="gemini-2.5-flash")

    def get_mapped_programmes(self) -> list:
        """Fetch all programs registered in the hierarchy, including unit counts."""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (f:Faculty)-[:HAS_DEPARTMENT]->(d:Department)-[:OFFERS_PROGRAM]->(p:Program)-[:HAS_UNIT]->(u:CourseUnit)
                RETURN f.name as faculty, d.name as department, p.name as name, count(u) as count
                ORDER BY faculty, department, name
                """
                r = session.run(query)
                return [{"name": row["name"], "department": row["department"], "faculty": row["faculty"], "count": row["count"]} for row in r]
        except Exception as e:
            log.error(f"Error fetching mapped programmes with hierarchies: {e}")
            return []

    def get_personalized_timetable(self, profile):
        """Returns a list of dicts for CSV/PDF export."""
        prog = profile.get("program") or profile.get("programme")
        year = profile.get("year") or profile.get("year_of_study")
        sem = profile.get("semester")
        
        if not prog: return []
        
        results = []
        with self.driver.session() as session:
            cypher = """
            MATCH (p:Program)-[r:HAS_UNIT]->(u:CourseUnit)
            WHERE toLower(p.name) CONTAINS toLower($prog)
            """
            params = {"prog": prog}
            if year:
                cypher += " AND r.year = $year"
                params["year"] = f"Year {year}"
            if sem:
                cypher += " AND r.semester = $sem"
                params["sem"] = str(sem)
            
            cypher += """
            OPTIONAL MATCH (u)-[:HAS_TIMESLOT]->(t:TimetableSlot)
            RETURN u.code as code, u.name as unit, t.day as day, t.time as time, t.room as room
            ORDER BY u.code, t.day_index, t.start_time
            """
            rows = session.run(cypher, **params).data()
            return rows

    def close(self):
        self.driver.close()
