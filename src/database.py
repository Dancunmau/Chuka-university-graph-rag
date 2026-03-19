import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL for production-ready data persistence as per the project proposal.
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL or not DATABASE_URL.startswith("postgresql"):
    raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string (starting with 'postgresql://')")

engine = create_engine(
    DATABASE_URL, 
    echo=False,
    pool_size=20,          # Standard number of persistent connections to keep open
    max_overflow=10,       # Allow bursting up to 30 connections during spikes
    pool_timeout=30,       # Wait max 30 seconds to get a connection before throwing timeout
    pool_recycle=1800      # Recycle connections every 30 minutes to drop dead/stale DB sockets
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

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

# Create tables
Base.metadata.create_all(bind=engine)

# add session_id column to existing table to prevent crashing
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE history ADD COLUMN session_id VARCHAR(100)"))
        conn.execute(text("CREATE INDEX ix_history_session_id ON history (session_id)"))
        conn.commit()
except Exception:
    pass # Column likely already exists

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

        # Return a plain dict — avoids DetachedInstanceError
        return {"user_id": user.user_id, "device_token": user.device_token}
    finally:
        db.close()

def save_user_profile(user_id, faculty, program, year_of_study, semester):
    """Save or update user's academic profile in PostgreSQL/SQLite"""
    db = SessionLocal()
    try:
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not profile:
            profile = UserProfile(user_id=user_id)
            db.add(profile)
            
        profile.faculty = faculty
        profile.program = program
        profile.year_of_study = int(year_of_study)
        profile.semester = int(semester)
        profile.updated_at = datetime.utcnow()
        db.commit()
    finally:
        db.close()

def log_chat_history(user_id, session_id, query_text, response_text):
    """Store the chat history for the user session"""
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
