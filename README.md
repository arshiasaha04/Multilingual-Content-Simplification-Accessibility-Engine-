# BhashaSaral: Multilingual Content Simplification & Accessibility Engine

An autonomous, self-correcting multilingual NLP and audio accessibility pipeline built with **LangGraph**, **FastAPI**, **Coqui TTS**, **OpenAI Whisper**, **Playwright**, and **Scikit-Learn**. 

Designed in alignment with **NEP 2020**, BhashaSaral translates and pedagogically simplifies complex STEM and educational content into regional Indian languages (**Hindi, Bengali, Tamil, Telugu**) across configurable reading levels while preventing the *"Sanskritization Trap"* (overly formal/archaic translations that alienate low-literacy learners).

## Key Architectural Innovations
1. **Diglossia-Aware Pedagogical Simplification & Technical Anchoring**: Adapts complex academic prose into natural spoken registers (*Bolchal ki Hindi*, *Chalit Bhasha* in Bengali) across 3 reading levels while preserving phonetic English technical terms alongside everyday analogies.
2. **Script-Aware Indic Readability Index (IRI)**: Custom morphological complexity evaluator built with **NumPy** inspecting Unicode Abugida blocks (`\u0900-\u097F`, `\u0980-\u09FF`) to quantify *Matra* (vowel diacritic) density, *Yuktakshar* (conjunct consonant / *Halant*) frequency, and average sentence length.
3. **Closed-Loop LangGraph Validation**: Automated quality verification using **Back-Translation**, **Scikit-Learn Char/Word N-Gram TF-IDF Cosine Similarity**, and **AI Semantic Consistency Checks**. Failed outputs automatically self-correct via conditional graph edges.
4. **Multimodal Ingestion & Neural Audio Accessibility**: Extracts clean web articles via headless **Playwright**, transcribes spoken lectures via **OpenAI Whisper**, and synthesizes pace-adjusted regional audio narration via **Coqui TTS** for low-literacy and visually impaired learners.
5. **SQL & Pandas/Seaborn Telemetry**: Tracks readability compression and semantic fidelity drift across runs in **SQLite/SQLAlchemy** and generates visual evaluation plots.

## Tech Stack
- **Orchestration & Backend**: Python, FastAPI, LangGraph, SQLAlchemy (SQL)
- **NLP, Speech & Scraping**: OpenAI Whisper, Coqui TTS, Playwright, Groq / Llama-3.3-70B
- **Validation & Data Science**: Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/multilingual-simplification-accessibility-engine.git
cd multilingual-simplification-accessibility-engine

# 2. Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium

# 3. Configure API Key
export GROQ_API_KEY="your_groq_api_key"  # On Windows PowerShell: $env:GROQ_API_KEY="your_key"

# 4. Run the FastAPI Server
uvicorn app.main:app --reload