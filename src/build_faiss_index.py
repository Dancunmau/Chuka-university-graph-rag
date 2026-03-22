"""
Overhauled to use pdfplumber for extraction and semantic chunking.
Targets:
  - Student Handbook 
  - CU Advert 
  - Timetable 
"""
import os, pickle, re, json
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import faiss
    import pdfplumber
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing missing dependencies")
    os.system("pip install sentence-transformers faiss-cpu pdfplumber tqdm -q")
    import faiss
    import pdfplumber
    from sentence_transformers import SentenceTransformer

BASE = 'd:/Jupyter notebook/Graph rag'
HANDBOOK_PATH = f'{BASE}/reports/Student-handbook-August-2024.docx.pdf'
ADVERT_PATH    = f'{BASE}/reports/CU-Advert-Jan-2024.pdf'
TIMETABLE_PDF  = f'{BASE}/reports/1st-Draft-Teaching-TT-Jan-April.-2026._compressed-1.pdf'

FAISS_INDEX_PATH = f'{BASE}/faiss_index.bin'
METADATA_PATH    = f'{BASE}/faiss_metadata.pkl'

def clean_text(text):
    if not text: return ""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.strip()

def is_toc_page(text: str) -> bool:
    """Detect Table of Contents pages to skip them from semantic indexing."""
    if not text:
        return False
    lines = text.split('\n')
    dot_leader_lines = sum(
        1 for line in lines
        if re.search(r'\.{4,}\s*\d+\s*$', line)                
        or re.search(r'\s{5,}\d{1,3}\s*$', line.rstrip())  
    )
    # If >30% of lines look like TOC entries, treat as TOC page
    return len(lines) > 3 and (dot_leader_lines / len(lines)) > 0.30

class SemanticChunker:
    """Helper to split text into overlapping windows while preserving context headers."""
    def __init__(self, chunk_size=800, overlap=150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text, source, page, base_header=""):
        if not text or len(text) < 50:
            return []

        # Simple semantic splitting by paragraph/sentence where possible
        paragraphs = re.split(r'\n\n+', text)
        chunks = []
        current_chunk = ""
        current_header = base_header

        for para in paragraphs:
            # Check for new headers (All caps or numbered)
            lines = para.split('\n')
            if lines and (re.match(r'^\d+\.\d+.*', lines[0].strip()) or (lines[0].strip().isupper() and len(lines[0].strip()) > 10)):
                current_header = lines[0].strip()

            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(self._format_chunk(current_chunk, source, page, current_header))
                
                # If paragraph itself is huge, split it
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_large_text(para)
                    for sc in sub_chunks[:-1]:
                        chunks.append(self._format_chunk(sc, source, page, current_header))
                    current_chunk = sub_chunks[-1] + "\n\n"
                else:
                    current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(self._format_chunk(current_chunk, source, page, current_header))
        
        return chunks

    def _split_large_text(self, text):
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.overlap)]

    def _format_chunk(self, text, source, page, header):
        display_header = header if header else "General Information"
        return {
            "source": source,
            "page": page,
            "header": display_header,
            "text": f"[{source} - {display_header}] (Page {page})\n{text.strip()}"
        }

def process_handbook():
    print(f"Processing Handbook: {HANDBOOK_PATH}")
    all_chunks = []
    chunker = SemanticChunker()
    current_header = "Table of Contents"

    with pdfplumber.open(HANDBOOK_PATH) as pdf:
        for page in tqdm(pdf.pages, desc="Handbook Pages"):
            text = page.extract_text()
            if not text: continue

            # Skip Table of Contents pages 
            if is_toc_page(text):
                print(f"Skipping TOC page {page.page_number}")
                continue

            # Detect chapter-level headers to propagate
            lines = text.split('\n')
            for line in lines[:5]:
                if re.match(r'^CHAPTER\s+\d+.*', line.strip().upper()):
                    current_header = line.strip()
                    break

            page_chunks = chunker.chunk(text, "Student Handbook", page.page_number, current_header)
            all_chunks.extend(page_chunks)
    return all_chunks

def process_advert():
    print(f"Processing Advert: {ADVERT_PATH}")
    all_chunks = []
    chunker = SemanticChunker(chunk_size=1200) 
    with pdfplumber.open(ADVERT_PATH) as pdf:
        current_level = "Academic Programmes"
        for page in tqdm(pdf.pages, desc="Advert Pages"):
            tables = page.extract_tables()
            page_text = page.extract_text()
            
            # Determine programme level
            if page_text:
                for l in ["BACHELOR", "DIPLOMA", "CERTIFICATE", "MASTER", "PhD"]:
                    if l in page_text.upper():
                        current_level = f"{l.capitalize()} Programmes"
                        break
            
            table_text = ""
            if tables:
                for table in tables:
                    for row in table:
                        if row:
                            table_text += " | ".join([str(c).replace('\n', ' ').strip() for c in row if c]) + "\n"
            
            combined = ""
            if page_text: combined += page_text + "\n"
            if table_text: combined += "\n--- Structured Programme Data ---\n" + table_text
            
            if combined.strip():
                page_chunks = chunker.chunk(combined, "CU Advert", page.page_number, current_level)
                all_chunks.extend(page_chunks)
    return all_chunks

def process_timetable():
    print(f"Processing Timetable: {TIMETABLE_PDF}")
    all_chunks = []
    if not os.path.exists(TIMETABLE_PDF):
        return []
    
    chunker = SemanticChunker(chunk_size=1500) 
    with pdfplumber.open(TIMETABLE_PDF) as pdf:
        current_day = "Schedule"
        for page in tqdm(pdf.pages, desc="Timetable Pages"):
            text = page.extract_text()
            tables = page.extract_tables()
            
            if text:
                for d in ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]:
                    if d in text.upper():
                        current_day = d
                        break
            
            table_text = ""
            if tables:
                for table in tables:
                    for row in table:
                        if row:
                            table_text += " | ".join([str(c).replace('\n', ' ').strip() for c in row if c]) + "\n"
            
            combined = ""
            if text: combined += text + "\n"
            if table_text: combined += "\n--- Daily Schedule Table ---\n" + table_text
            
            if combined.strip():
                page_chunks = chunker.chunk(combined, "Teaching Timetable", page.page_number, current_day)
                all_chunks.extend(page_chunks)
    return all_chunks

def build_index():
    # 1. Collect all chunks
    all_chunks = []
    all_chunks.extend(process_handbook())
    all_chunks.extend(process_advert())
    all_chunks.extend(process_timetable())
    
    # 2. Add existing repository items if the index exists
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'rb') as f:
            old_metadata = pickle.load(f)
        # Filter out old PDF chunks
        repo_chunks = [c for c in old_metadata if c['source'].startswith('Repository:')]
        print(f"Keeping {len(repo_chunks)} repository items from existing index.")
        all_chunks.extend(repo_chunks)

    print(f"Total chunks to index: {len(all_chunks)}")

    # 3. Embed
    print("Loading sentence-transformers model")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    texts = [c['text'] for c in all_chunks]

    print("Generating embeddings")
    vecs = embedder.encode(texts, batch_size=32, show_progress_bar=True).astype(np.float32)

    # Normalize
    faiss.normalize_L2(vecs)

    # 4. Build Index
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # 5. Save
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(all_chunks, f)

    print(f"\nFAISS index saved: {index.ntotal} vectors")
    print(f"Index: {FAISS_INDEX_PATH}")
    print(f"Metadata: {METADATA_PATH}")

if __name__ == "__main__":
    build_index()
