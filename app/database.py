from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import Config

engine = create_engine(Config.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SimplificationJob(Base):
    __tablename__ = "simplification_jobs"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(50))               # 'text', 'url', 'audio'
    original_text = Column(Text)
    simplified_english = Column(Text)
    translated_indic_text = Column(Text)
    back_translated_text = Column(Text)
    target_language = Column(String(10))           # 'hi', 'bn', 'ta', 'te'
    reading_level = Column(String(20))             # 'level_1', 'level_2', 'level_3'
    
    # Quality Validation Telemetry
    source_flesch_kincaid = Column(Float)
    simplified_flesch_kincaid = Column(Float)
    indic_readability_score = Column(Float)        # Custom Matra & Conjunct index (0-100, lower is simpler)
    back_translation_cosine_sim = Column(Float)    # Scikit-learn TF-IDF / Char N-gram Cosine Similarity
    llm_semantic_consistency = Column(Float)       # AI Semantic Verification (0.0 to 1.0)
    composite_quality_score = Column(Float)
    iterations_taken = Column(Integer)             # Number of LangGraph self-correction loops
    
    audio_file_path = Column(String(255), nullable=True)
    glossary_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)