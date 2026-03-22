import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import database


def test_normalize_postgres_scheme():
    raw_url = "postgres://user:password@localhost:5432/chuka_db"
    normalized = database._normalize_database_url(raw_url)
    assert normalized == "postgresql://user:password@localhost:5432/chuka_db"


def test_sqlite_fallback_path_points_to_data_dir():
    sqlite_url = database._default_sqlite_url()
    assert sqlite_url.startswith("sqlite:///")
    assert "data/chuka_graphrag.db" in sqlite_url.replace("\\", "/")


def test_database_configuration_exposes_supported_backend():
    assert database.DATABASE_BACKEND in {"postgresql", "sqlite"}
    assert database.DATABASE_URL
    if "DATABASE_URL" not in os.environ:
        assert database.DATABASE_BACKEND == "sqlite"
