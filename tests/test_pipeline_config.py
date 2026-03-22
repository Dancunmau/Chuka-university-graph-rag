import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_pipeline_reads_gemini_keys_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "primary-key")
    monkeypatch.setenv("GEMINI_API_KEY2", "backup-key")
    monkeypatch.setenv("GEMINI_API_KEY3", "")
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://example.databases.neo4j.io")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "password")

    import src.chuka_graphrag_pipeline as pipeline

    pipeline = importlib.reload(pipeline)

    assert pipeline.GEMINI_API_KEY == "primary-key"
    assert pipeline.GEMINI_KEYS == ["primary-key", "backup-key"]
    assert pipeline.NEO4J_URI == "neo4j+s://example.databases.neo4j.io"
