import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

# We use PostgreSQL for production-ready data persistence as per the project proposal.
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL or not DATABASE_URL.startswith("postgresql"):
    raise ValueError("DATABASE_URL must be a valid PostgreSQL connection string (starting with 'postgresql://')")

engine = create_engine(DATABASE_URL, echo=False)
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
    query_text = Column(Text)
    response_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="history")

# Create tables
Base.metadata.create_all(bind=engine)

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

        # Return a plain dict — avoids DetachedInstanceError entirely
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

def log_chat_history(user_id, query_text, response_text):
    """Store the chat history for the user session"""
    db = SessionLocal()
    try:
        chat = History(
            user_id=user_id,
            query_text=query_text,
            response_text=response_text
        )
        db.add(chat)
        db.commit()
    finally:
        db.close()

def get_chat_history(user_id, limit=50):
    """Fetch previous chat history for UI rendering"""
    db = SessionLocal()
    try:
        history = db.query(History).filter(History.user_id == user_id).order_by(History.timestamp.asc()).limit(limit).all()
        return history
    finally:
        db.close()

def clear_chat_history(user_id):
    """Delete all chat history records for a specific user ID"""
    db = SessionLocal()
    try:
        db.query(History).filter(History.user_id == user_id).delete()
        db.commit()
    finally:
        db.close()
