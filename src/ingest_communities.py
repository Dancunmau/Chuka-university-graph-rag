"""
Hybrid ingestion of communities.csv:
  1. Neo4j: Create RepositoryItem nodes linked to CourseUnit via extracted course code
  2. FAISS: Append community titles to the existing semantic index
"""
import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv('d:/Jupyter notebook/Graph rag/.env')

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PWD  = os.getenv("NEO4J_PASSWORD")

BASE = 'd:/Jupyter notebook/Graph rag'
COMMUNITIES_PATH = f'{BASE}/data/communities.csv'
FAISS_INDEX_PATH = f'{BASE}/faiss_index.bin'
METADATA_PATH    = f'{BASE}/faiss_metadata.pkl'

# Communities relevant to students skip Archives & Corporate Docs
RELEVANT_COMMUNITIES = {
    'Books/Book Chapters/Book Reviews',
    'Conferences',
    'Journals/Journal Articles',
    'Theses and Dissertations',
    'Research Articles',
    'Student Projects',
}

def extract_course_codes(title):
    """Extract course codes like COSC 121 or SOCI 403 from a title string."""
    matches = re.findall(r'[A-Z]{2,5}\s*\d{3,4}', str(title))
    return list(set(re.sub(r'([A-Z]+)\s*(\d+)', r'\1 \2', c) for c in matches))


def ingest_to_neo4j(df, driver):
    """Create RepositoryItem nodes and link them to CourseUnit nodes."""
    items_with_codes = []
    items_without_codes = []

    for _, row in df.iterrows():
        codes = extract_course_codes(row['title'])
        entry = {
            'link': row['repository_link'],
            'title': row['title'],
            'author': row.get('author', ''),
            'type': row.get('type', ''),
            'year': str(row.get('year', '')),
            'community': row['community'],
            'codes': codes,
        }
        if codes:
            items_with_codes.append(entry)
        else:
            items_without_codes.append(entry)

    print(f"  Items with course codes: {len(items_with_codes)}")
    print(f"  Items without course codes (general content): {len(items_without_codes)}")
    all_items = items_with_codes + items_without_codes

    with driver.session() as session:
        # Constraint
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:RepositoryItem) REQUIRE r.link IS UNIQUE")

        batch_size = 300
        total_batches = -(-len(all_items) // batch_size)
        for i in range(0, len(all_items), batch_size):
            batch = all_items[i:i+batch_size]
            b_num = i // batch_size + 1

            # Step A: Create/merge RepositoryItem nodes
            session.run("""
            UNWIND $data AS row
            MERGE (r:RepositoryItem {link: row.link})
            SET r.title     = row.title,
                r.author    = row.author,
                r.type      = row.type,
                r.year      = row.year,
                r.community = row.community
            """, data=batch)

            # Step B: For items with codes, create/link CourseUnit nodes
            coded = [
                {'link': e['link'], 'code': code}
                for e in batch
                for code in e['codes']
            ]
            if coded:
                session.run("""
                UNWIND $data AS row
                MATCH (r:RepositoryItem {link: row.link})
                MERGE (c:CourseUnit {code: row.code})
                MERGE (c)-[:HAS_RESOURCE]->(r)
                """, data=coded)

            print(f"  Neo4j batch {b_num}/{total_batches} done.")


def ingest_to_faiss(df):
    """Append community items as chunks to the existing FAISS index."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("faiss / sentence-transformers not installed.")
        return

    # Load existing index and metadata
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        print(f"  Loaded existing FAISS index with {index.ntotal} vectors.")
    else:
        print("  No existing FAISS index found — creating a new one.")
        index = None
        metadata = []

    # Build text chunks from the community data
    new_chunks = []
    for _, row in df.iterrows():
        text = f"{row['title']}"
        if row.get('author'):
            text += f" — {row['author']}"
        if row.get('type'):
            text += f" ({row['type']})"
        if row.get('year'):
            text += f" [{row['year']}]"
        new_chunks.append({
            'source': f"Repository:{row['community']}",
            'page': '—',
            'text': text,
            'link': row['repository_link'],
        })

    print(f"  Encoding {len(new_chunks)} community items into FAISS...")
    embedder = SentenceTransformer("all-mpnet-base-v2")
    texts = [c['text'] for c in new_chunks]
    vecs = embedder.encode(texts, batch_size=64, show_progress_bar=True).astype(np.float32)
    faiss.normalize_L2(vecs)

    if index is None:
        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)

    index.add(vecs)
    metadata.extend(new_chunks)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"  FAISS index updated: {index.ntotal} total vectors.")


def main():
    df = pd.read_csv(COMMUNITIES_PATH).fillna('')

    print(f"Total rows in communities.csv: {len(df)}")
    print(f"Communities found: {df['community'].unique().tolist()}\n")

    # Filter to student relevant communities
    relevant_df = df[df['community'].isin(RELEVANT_COMMUNITIES)]
    print(f"Rows after filtering to relevant communities: {len(relevant_df)}\n")

    # Neo4j 
    print("Step 1: Ingesting into Neo4j...")
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    ingest_to_neo4j(relevant_df, driver)
    driver.close()
    print("Neo4j ingestion complete.\n")

    # FAISS 
    print("Step 2: Updating FAISS index...")
    ingest_to_faiss(relevant_df)
    print("FAISS update complete.\n")

    print("All done!")


if __name__ == "__main__":
    main()
