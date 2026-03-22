"""
Available tools:
    graph_stats     - Print Neo4j node counts and relationship types
    communities     - Verify RepositoryItem linkage and sample SOCI resources
    unit_mapping    - Check unit→programme mappings and orphan count
    graph_links     - Spot-check past paper linkages in the graph
    faiss_search    - Keyword-search the FAISS metadata (use with --query)
    gemini          - Test Gemini API key and list available models
    pipeline        - Full pipeline smoke test with sample queries
    pdf_rag         - PDF-specific RAG verification (handbook, advert, timetable)
    all             - Run all tools (skips pipeline/pdf_rag to avoid API cost)

Examples:
    python dev_tools.py --tool graph_stats
    python dev_tools.py --tool faiss_search --query fee
    python dev_tools.py --tool gemini
    python dev_tools.py --tool all
"""

import os
import sys
import pickle
import argparse
from dotenv import load_dotenv

# Environment
BASE = 'd:/Jupyter notebook/Graph rag'
load_dotenv(os.path.join(BASE, '.env'))
sys.path.insert(0, BASE)

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PWD  = os.getenv("NEO4J_PASSWORD")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
METADATA_PATH = os.path.join(BASE, 'faiss_metadata.pkl')

DIVIDER = "=" * 60


def _neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))


# 1. Graph Stats
def tool_graph_stats():
    """Print node counts and relationship types from Neo4j."""
    print(f"\n{DIVIDER}")
    print("TOOL: graph_stats — Neo4j Overview")
    print(DIVIDER)
    driver = _neo4j_driver()
    with driver.session() as s:
        print("\n--- Node Counts ---")
        for r in s.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC"):
            print(f"  {r['label']}: {r['count']}")

        print("\n--- Relationship Types ---")
        for r in s.run("CALL db.relationshipTypes()"):
            print(f"  {r[0]}")

        print("\n--- Relationship Counts ---")
        for r in s.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS count ORDER BY count DESC"):
            print(f"  {r['rel']}: {r['count']}")
    driver.close()
    print("\n graph_stats complete.")


# 2. Communities / Repository
def tool_communities():
    """Verify RepositoryItem counts and sample SOCI-linked resources."""
    print(f"\n{DIVIDER}")
    print("TOOL: communities — Repository & Community Linkage")
    print(DIVIDER)
    driver = _neo4j_driver()
    with driver.session() as s:
        total  = s.run("MATCH (r:RepositoryItem) RETURN count(r) AS n").single()["n"]
        linked = s.run("MATCH (c:CourseUnit)-[:HAS_RESOURCE]->(r:RepositoryItem) "
                       "RETURN count(distinct r) AS n").single()["n"]
        soci   = s.run("MATCH (c:CourseUnit)-[:HAS_RESOURCE]->(r:RepositoryItem) "
                       "WHERE c.code STARTS WITH 'SOCI' "
                       "RETURN c.code AS code, r.title AS title LIMIT 5").data()

        print(f"\nTotal RepositoryItems : {total}")
        print(f"Linked to CourseUnit  : {linked}")
        print(f"Unlinked              : {total - linked}")
        print("\nSample SOCI resources:")
        for row in soci:
            print(f"  {row['code']}: {row['title']}")
    driver.close()
    print("\ncommunities complete.")


# 3. Unit Mapping
def tool_unit_mapping():
    """Verify unit→programme mappings and count orphan units."""
    print(f"\n{DIVIDER}")
    print("TOOL: unit_mapping — Unit→Programme Mappings")
    print(DIVIDER)
    driver = _neo4j_driver()
    with driver.session() as s:
        print("\n--- Sample Units Linked to Programmes ---")
        for r in s.run("MATCH (p:Programme)-[r:HAS_UNIT]->(u:CourseUnit) "
                       "RETURN p.name AS Programme, u.code AS Unit, r.year AS Year, r.semester AS Sem "
                       "LIMIT 5"):
            print(f"   {r['Unit']} → {r['Programme']} [Year {r['Year']}, Sem {r['Sem']}]")

        print("\n--- HAS_UNIT / HAS_TIMESLOT Relationship Spot-Check ---")
        for r in s.run("MATCH (p:Programme)-[:HAS_UNIT]->(u:CourseUnit) RETURN p.name, count(u) LIMIT 3"):
            print(f"  Prog: {r[0]} | Units: {r[1]}")
        for r in s.run("MATCH (u:CourseUnit)-[:HAS_TIMESLOT]->(t:TimetableSlot) RETURN u.code, count(t) LIMIT 3"):
            print(f"  Unit: {r[0]} | Slots: {r[1]}")

        mapping_count = s.run("MATCH ()-[r:HAS_UNIT]->() RETURN count(r) AS n").single()["n"]
        orphan_count  = s.run("MATCH (u:CourseUnit) WHERE NOT (u)<-[:HAS_UNIT]-(:Programme) "
                              "RETURN count(u) AS n").single()["n"]
        print(f"\nSummary:")
        print(f"  HAS_UNIT mappings total : {mapping_count}")
        print(f"  Orphan units (no prog)  : {orphan_count}")
    driver.close()
    print("\n unit_mapping complete.")


# 4. Graph Links
def tool_graph_links():
    """Spot-check past paper linkages in the graph."""
    print(f"\n{DIVIDER}")
    print("TOOL: graph_links — Past Paper Linkage Spot-Check")
    print(DIVIDER)
    driver = _neo4j_driver()
    with driver.session() as s:
        print("\n--- 'SOCIAL IMPLICATIONS' Past Paper Links ---")
        rows = s.run("MATCH (c:CourseUnit)-[:HAS_PAST_PAPER]->(p:PastPaper) "
                     "WHERE p.title CONTAINS 'SOCIAL IMPLICATIONS' "
                     "RETURN c.code, p.title").data()
        if rows:
            for r in rows:
                print(f"  {r['c.code']} → {r['p.title']}")
        else:
            print("  (none found)")

        print("\n--- COSC 471 Past Paper Links ---")
        rows2 = s.run("MATCH (c:CourseUnit)-[:HAS_PAST_PAPER]->(p:PastPaper) "
                      "WHERE c.code = 'COSC 471' RETURN c.code, p.title").data()
        if rows2:
            for r in rows2:
                print(f"  {r['c.code']} → {r['p.title']}")
        else:
            print("  (none found)")

        total_papers = s.run("MATCH ()-[:HAS_PAST_PAPER]->() RETURN count(*) AS n").single()["n"]
        print(f"\nTotal HAS_PAST_PAPER edges: {total_papers}")
    driver.close()
    print("\n graph_links complete.")


# 5. FAISS Metadata Search
def tool_faiss_search(query: str):
    """Keyword-search the FAISS metadata chunks for a given term."""
    print(f"\n{DIVIDER}")
    print(f"TOOL: faiss_search — Searching for '{query}'")
    print(DIVIDER)
    if not os.path.exists(METADATA_PATH):
        print(f" Metadata file not found: {METADATA_PATH}")
        return

    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)

    matches = [m for m in metadata if query.lower() in m['text'].lower()]
    print(f"\nSearched {len(metadata)} chunks — found {len(matches)} matches.")

    for i, m in enumerate(matches[:20]):
        text = m['text']
        idx  = text.lower().find(query.lower())
        start, end = max(0, idx - 100), min(len(text), idx + 350)
        print(f"\n--- Match {i+1} ---")
        print(f"  Source : {m['source']} (Page {m['page']})")
        print(f"  Header : {m.get('header', 'N/A')}")
        print(f"  Context: ...{text[start:end]}...")

    if len(matches) > 20:
        print(f"\n  (showing 20 of {len(matches)} matches)")
    print("\n faiss_search complete.")


# 6. Gemini API Test
def tool_gemini():
    """Check Gemini API key and find a working model."""
    print(f"\n{DIVIDER}")
    print("TOOL: gemini — Gemini API Key & Model Check")
    print(DIVIDER)
    if not GEMINI_KEY:
        print(" GEMINI_API_KEY not found in .env")
        return

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_KEY)

    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        print(f"\nAvailable models ({len(models)} total): {', '.join(models[:5])}{'...' if len(models) > 5 else ''}")
    except Exception as e:
        print(f"Could not list models: {e}")
        models = ["gemini-2.0-flash", "gemini-1.5-flash"]

    print("\nTesting first working model...")
    for name in models:
        clean = name.replace("models/", "")
        try:
            resp = genai.GenerativeModel(clean).generate_content("Say 'OK'")
            print(f" [{clean}] → {resp.text.strip()}")
            break
        except Exception as e:
            tag = "Quota exceeded" if ("429" in str(e) or "quota" in str(e).lower()) else str(e)[:80]
            print(f" [{clean}] → {tag}")
    print("\n gemini complete.")


# 7. Pipeline Smoke Test
def tool_pipeline():
    """Run full GraphRAG pipeline smoke test with sample academic queries."""
    print(f"\n{DIVIDER}")
    print("TOOL: pipeline — Full Pipeline Smoke Test")
    print(DIVIDER)
    os.chdir(BASE)
    from chuka_graphrag_pipeline import GraphRAGAssistant

    assistant = GraphRAGAssistant()
    print("Assistant initialised.\n")

    profile = {
        "faculty": "Faculty of Science and Technology",
        "program": "BSc Computer Science",
        "year": "1",
        "semester": "1"
    }

    tests = [
        "What units do I take in Year 1 Semester 1 of Computer Science?",
        "Find past papers for COSC 121",
        "Give me past papers for data mining",
        "What are the exam regulations in the student handbook?",
    ]

    for q in tests:
        print(f"Q: {q}")
        ans = assistant.generate_response(q, profile)
        print(f"A: {ans[:400]}...\n{'-'*50}")

    assistant.close()
    print("\npipeline complete.")


# 8. PDF RAG Verification
def tool_pdf_rag():
    """Run PDF-specific RAG queries to verify handbook, advert, and timetable retrieval."""
    print(f"\n{DIVIDER}")
    print("TOOL: pdf_rag — PDF RAG Verification")
    print(DIVIDER)
    from chuka_graphrag_pipeline import GraphRAGAssistant

    assistant = GraphRAGAssistant()
    profile = {
        "faculty": "Science and Technology",
        "program": "BSc Computer Science",
        "year": "1",
        "semester": "1"
    }

    tests = [
        "What are the rules for special exams in the handbook?",
        "What is the tuition fee for BSc Computer Science in the advert?",
        "Show me the timetable for COSC 121 on Monday",
    ]

    for q in tests:
        print(f"\nQUERY: {q}")
        print(f"RESPONSE:\n{assistant.generate_response(q, profile)}")
        print("-" * 50)

    assistant.close()
    print("\n pdf_rag complete.")


# CLI Entry Point
TOOLS = {
    "graph_stats" : tool_graph_stats,
    "communities" : tool_communities,
    "unit_mapping": tool_unit_mapping,
    "graph_links" : tool_graph_links,
    "faiss_search": tool_faiss_search,
    "gemini"      : tool_gemini,
    "pipeline"    : tool_pipeline,
    "pdf_rag"     : tool_pdf_rag,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chuka GraphRAG — Developer Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:<16} {v.__doc__}" for k, v in TOOLS.items())
    )
    parser.add_argument("--tool",  required=True, choices=list(TOOLS.keys()) + ["all"],
                        help="Which diagnostic tool to run")
    parser.add_argument("--query", default="fee",
                        help="Search term for faiss_search (default: 'fee')")
    args = parser.parse_args()

    if args.tool == "all":
        # Skip pipeline and pdf_rag in bulk-run to avoid API cost
        for name in ["graph_stats", "communities", "unit_mapping", "graph_links", "faiss_search", "gemini"]:
            if name == "faiss_search":
                tool_faiss_search(args.query)
            else:
                TOOLS[name]()
        print(f"\n{DIVIDER}")
        print("All non-API tools complete. Run --tool pipeline or --tool pdf_rag separately.")
        print(DIVIDER)
    elif args.tool == "faiss_search":
        tool_faiss_search(args.query)
    else:
        TOOLS[args.tool]()
