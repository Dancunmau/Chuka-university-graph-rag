"""
neo4j_utils.py — Shared utilities for the Chuka University GraphRAG ingestion pipeline.

Exports:
    get_driver()       — loads .env and returns a connected Neo4j driver
    get_project_root() — returns the absolute project root (parent of /src)
    run_batch()        — runs a Cypher UNWIND query in batches with progress
    close_driver()     — safely closes a driver
    tc()               — converts a string to proper English Title Case
"""

import re
import os
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv


# Words that should remain lowercase in title case (unless first word)
_LOWERCASE_WORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so",
    "at", "by", "in", "of", "on", "to", "up", "as", "is",
}


def tc(value: str) -> str:
    """
    Convert a string to English Title Case.
    Short connecting words (of, and, the, in...) stay lowercase unless first word.

    Example:
        "FACULTY OF SCIENCE & TECHNOLOGY"  ->  "Faculty of Science & Technology"
        "department of computer sciences"   ->  "Department of Computer Sciences"
    """
    if not value:
        return value
    words = re.split(r'(\s+)', value.strip())
    result = []
    first = True
    for token in words:
        if token.strip() == "":
            result.append(token)
            continue
        low = token.lower()
        if first or low not in _LOWERCASE_WORDS:
            result.append(token.capitalize())
        else:
            result.append(low)
        first = False
    return "".join(result)


def get_project_root() -> Path:
    """Return the absolute path to the project root (parent of /src)."""
    return Path(__file__).resolve().parent.parent


def get_driver():
    """Load .env from project root and return a connected Neo4j driver."""
    env_path = get_project_root() / ".env"
    load_dotenv(dotenv_path=env_path, override=False)

    uri  = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd  = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, pwd]):
        raise EnvironmentError(
            "NEO4J_URI, NEO4J_USERNAME or NEO4J_PASSWORD not set. "
            f"Checked .env at: {env_path}"
        )

    return GraphDatabase.driver(uri, auth=(user, pwd))


def run_batch(session, query: str, data: list, batch_size: int = 500,
              label: str = "rows") -> int:
    """Execute a Cypher UNWIND query in chunks. Returns total rows processed."""
    total = len(data)
    num_batches = -(-total // batch_size)
    processed = 0
    for i in range(0, total, batch_size):
        batch = data[i: i + batch_size]
        session.run(query, data=batch)
        processed += len(batch)
        print(f"  [{i // batch_size + 1}/{num_batches}] {processed}/{total} {label}")
    return processed


def close_driver(driver) -> None:
    """Safely close a Neo4j driver."""
    try:
        driver.close()
    except Exception:
        pass
