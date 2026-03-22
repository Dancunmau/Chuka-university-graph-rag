"""Persistence layer for the Chuka University GraphRAG Assistant.
Orchestrates SQLAlchemy-backed connections for user state and chat history.
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit may be unavailable in some test contexts.
    st = None

load_dotenv()

log = logging.getLogger(__name__)


def _get_streamlit_secret(key):
    """Safely read a Streamlit secret when running inside Streamlit Cloud."""
    if st is None:
        return None

    try:
        return st.secrets.get(key)
    except Exception:
        return None


def _normalize_database_url(raw_url):
    """Normalize provider-specific database URL variants."""
    if not raw_url:
        return None

    normalized = raw_url.strip()
    if normalized.startswith("postgres://"):
        return "postgresql://" + normalized[len("postgres://") :]
    return normalized


def _default_sqlite_url():
    """Return a local SQLite path used when Postgres is not configured."""
    db_dir = Path(__file__).resolve().parent.parent / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "chuka_graphrag.db"
    return f"sqlite:///{db_path.as_posix()}"


def _resolve_database_url():
    """Pick the best available database URL for the current runtime."""
    raw_url = os.getenv("DATABASE_URL") or _get_streamlit_secret("DATABASE_URL")
    database_url = _normalize_database_url(raw_url)

    if database_url and database_url.startswith("postgresql"):
        return database_url, "postgresql", False

    if database_url and database_url.startswith("sqlite"):
        return database_url, "sqlite", False

    fallback_url = _default_sqlite_url()
    if database_url:
        log.warning(
            "Unsupported DATABASE_URL scheme '%s'. Falling back to SQLite at %s.",
            database_url.split(":", 1)[0],
            fallback_url,
        )
    else:
        log.warning(
            "DATABASE_URL is not configured. Falling back to SQLite at %s.",
            fallback_url,
        )
    return fallback_url, "sqlite", True


# Database configuration
DATABASE_URL, DATABASE_BACKEND, USING_FALLBACK_DATABASE = _resolve_database_url()
DATABASE_STATUS_MESSAGE = (
    "Using local SQLite fallback because DATABASE_URL is missing or unsupported. "
    "Chat history and profiles may reset when the app restarts."
    if USING_FALLBACK_DATABASE
    else f"Using configured {DATABASE_BACKEND} database backend."
)

engine_kwargs = {"echo": False}
if DATABASE_BACKEND == "postgresql":
    engine_kwargs.update(
        pool_pre_ping=True,
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
    )
else:
    engine_kwargs.update(connect_args={"check_same_thread": False})

engine = create_engine(
    DATABASE_URL,
    **engine_kwargs,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Schema Definitions (ORMs)
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    device_token = Column(String(200), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    history = relationship("History", back_populates="user")

class UserProfile(Base):
    __tablename__ = "user_profile"
    
    profile_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    faculty = Column(String(120))
    department = Column(String(120))
    program = Column(String(200))
    year_of_study = Column(Integer)
    semester = Column(Integer)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = relationship("User", back_populates="profile")

class History(Base):
    __tablename__ = "history"
    
    history_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    session_id = Column(String(100), index=True, nullable=True)
    query_text = Column(Text)
    response_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="history")


# Automated Schema Migrations / Initialization
Base.metadata.create_all(bind=engine)

# Migration: Provision 'session_id' for multi-session chat tracking.
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE history ADD COLUMN session_id VARCHAR(100)"))
        conn.execute(text("CREATE INDEX ix_history_session_id ON history (session_id)"))
        conn.commit()
except Exception:
    pass 

# Migration: Provision 'department' column for granular onboarding profiles.
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE user_profile ADD COLUMN department VARCHAR(120)"))
        conn.commit()
except Exception:
    pass 

# Data Access Objects (DAOs)
def get_or_create_user(device_token=None):
    """Retrieve user by token, or create a new one. Returns a plain dict."""
    db = SessionLocal()
    try:
        user = None
        if device_token:
            user = db.query(User).filter(User.device_token == device_token).first()
            
        if not user:
            new_token = device_token or str(uuid.uuid4())
            user = User(device_token=new_token)
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            user.last_seen = datetime.utcnow()
            db.commit()
            db.refresh(user)

        # Return a plain dict to avoid error
        return {"user_id": user.user_id, "device_token": user.device_token}
    finally:
        db.close()

def save_user_profile(user_id, faculty, department, program, year_of_study, semester):
    """Save or update user's academic profile in the PostgreSQL database."""
    db = SessionLocal()
    try:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile:
            profile = UserProfile(user_id=user_id)
            db.add(profile)
            
        profile.faculty = faculty
        profile.department = department
        profile.program = program
        profile.year_of_study = int(year_of_study)
        profile.semester = int(semester)
        profile.updated_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()

def log_chat_history(user_id, session_id, query_text, response_text):
    """Store the chat history for the user session."""
    db = SessionLocal()
    try:
        chat = History(
            user_id=user_id,
            session_id=session_id,
            query_text=query_text,
            response_text=response_text
        )
        db.add(chat)
        db.commit()
    finally:
        db.close()

def get_chat_history(user_id, session_id=None, limit=100):
    """Fetch previous chat history for UI rendering"""
    db = SessionLocal()
    try:
        query = db.query(History).filter(History.user_id == user_id)
        if session_id:
            if session_id == "default":
                query = query.filter(History.session_id.is_(None))
            else:
                query = query.filter(History.session_id == session_id)
        history = query.order_by(History.timestamp.asc()).limit(limit).all()
        return history
    finally:
        db.close()

def get_user_sessions(user_id):
    """Get all distinct chat sessions for the sidebar"""
    db = SessionLocal()
    try:
        history = db.query(History).filter(History.user_id == user_id).order_by(History.timestamp.asc()).all()
        sessions = {}
        for h in history:
            sid = h.session_id or "default"
            if sid not in sessions:
                title = h.query_text[:30] + "..." if len(h.query_text) > 30 else h.query_text
                sessions[sid] = {"session_id": sid, "title": title, "timestamp": h.timestamp}
        
        # sort by timestamp desc
        return sorted(sessions.values(), key=lambda x: x["timestamp"], reverse=True)
    finally:
        db.close()

def clear_chat_history(user_id, session_id=None):
    """Delete all chat history records or a specific session"""
    db = SessionLocal()
    try:
        q = db.query(History).filter(History.user_id == user_id)
        if session_id:
            q = q.filter(History.session_id == session_id)
        q.delete()
        db.commit()
    finally:
        db.close()
