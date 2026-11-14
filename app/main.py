import json
from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import init_db, SessionLocal, SimplificationJob
from app.pipeline_graph import MultilingualSimplificationGraph
from app.ingestion import ContentIngestionEngine
from app.tts_engine import NeuralAccessibilityAudioEngine
from app.analytics import TelemetryAnalyticsGenerator

init_db()

app = FastAPI(
    title="BhashaSaral: Multilingual Content Simplification & Accessibility Engine",
    description="LangGraph + Coqui TTS + Whisper + Playwright NLP Engine for Accessible Regional Education",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

pipeline_engine = MultilingualSimplificationGraph()
ingestion_engine = ContentIngestionEngine()
tts_engine = NeuralAccessibilityAudioEngine()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class TextSimplifyRequest(BaseModel):
    text: str
    target_language: str = "hi"
    reading_level: str = "level_1"
    generate_audio: bool = True

class URLSimplifyRequest(BaseModel):
    url: str
    target_language: str = "bn"
    reading_level: str = "level_1"
    generate_audio: bool = True

def _execute_and_persist(
    text: str,
    source_type: str,
    target_language: str,
    reading_level: str,
    generate_audio: bool,
    db: Session
) -> dict:
    state = pipeline_engine.run_pipeline(
        text=text,
        target_language=target_language,
        reading_level=reading_level
    )

    audio_url = None
    if generate_audio and state["translated_indic_text"]:
        audio_url = tts_engine.synthesize_narration(
            text=state["translated_indic_text"],
            lang_code=target_language,
            reading_level=reading_level
        )

    job = SimplificationJob(
        source_type=source_type,
        original_text=text,
        simplified_english=state["simplified_english"],
        translated_indic_text=state["translated_indic_text"],
        back_translated_text=state["back_translated_text"],
        target_language=target_language,
        reading_level=reading_level,
        source_flesch_kincaid=state["source_fk_grade"],
        simplified_flesch_kincaid=state["simplified_fk_grade"],
        indic_readability_score=state["indic_readability_score"],
        back_translation_cosine_sim=state["back_translation_sim"],
        llm_semantic_consistency=state["llm_semantic_score"],
        composite_quality_score=state["composite_quality_score"],
        iterations_taken=state["iteration_count"],
        audio_file_path=audio_url,
        glossary_json=json.dumps(state["technical_glossary"], ensure_ascii=False)
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return {
        "job_id": job.id,
        "original_text": text,
        "simplified_english": state["simplified_english"],
        "translated_indic_text": state["translated_indic_text"],
        "back_translated_text": state["back_translated_text"],
        "technical_glossary": state["technical_glossary"],
        "quality_validation": {
            "source_flesch_kincaid_grade": state["source_fk_grade"],
            "simplified_flesch_kincaid_grade": state["simplified_fk_grade"],
            "indic_readability_index": state["indic_readability_score"],
            "indic_script_breakdown": state["indic_metrics_detail"],
            "back_translation_cosine_sim": state["back_translation_sim"],
            "ai_semantic_consistency": state["llm_semantic_score"],
            "composite_quality_score": state["composite_quality_score"],
            "langgraph_iterations": state["iteration_count"]
        },
        "audio_narration_url": audio_url
    }

@app.get("/")
def serve_dashboard():
    return FileResponse("static/index.html")

@app.post("/api/simplify/text")
def simplify_text_endpoint(req: TextSimplifyRequest, db: Session = Depends(get_db)):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    return _execute_and_persist(
        text=req.text,
        source_type="text",
        target_language=req.target_language,
        reading_level=req.reading_level,
        generate_audio=req.generate_audio,
        db=db
    )

@app.post("/api/simplify/url")
async def simplify_url_endpoint(req: URLSimplifyRequest, db: Session = Depends(get_db)):
    scraped_text = await ingestion_engine.extract_from_url(req.url)
    return _execute_and_persist(
        text=scraped_text,
        source_type="url",
        target_language=req.target_language,
        reading_level=req.reading_level,
        generate_audio=req.generate_audio,
        db=db
    )

@app.post("/api/simplify/audio")
async def simplify_audio_lecture_endpoint(
    file: UploadFile = File(...),
    target_language: str = Form("hi"),
    reading_level: str = Form("level_1"),
    db: Session = Depends(get_db)
):
    audio_bytes = await file.read()
    transcribed_text = ingestion_engine.transcribe_audio_lecture(audio_bytes, file.filename)
    return _execute_and_persist(
        text=transcribed_text,
        source_type="audio_whisper",
        target_language=target_language,
        reading_level=reading_level,
        generate_audio=True,
        db=db
    )

@app.get("/api/analytics/report")
def get_analytics_report(db: Session = Depends(get_db)):
    return TelemetryAnalyticsGenerator.generate_evaluation_charts(db)