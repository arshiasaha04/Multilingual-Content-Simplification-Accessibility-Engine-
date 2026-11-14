import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Supports Groq (Llama-3.3-70B) or OpenAI (GPT-4o-mini)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    
    DATABASE_URL = "sqlite:///./accessibility_engine.db"
    AUDIO_OUTPUT_DIR = os.path.join("static", "audio")
    PLOTS_OUTPUT_DIR = os.path.join("static", "plots")

    # Supported Indian Languages & Unicode Script Ranges
    SUPPORTED_LANGUAGES = {
        "hi": {
            "name": "Hindi (हिन्दी)",
            "script": "Devanagari",
            "unicode_range": (0x0900, 0x097F),
            "halant": "\u094d",
            "matras": [chr(c) for c in range(0x093E, 0x094D)],
            "tts_lang": "hi"
        },
        "bn": {
            "name": "Bengali (বাংলা)",
            "script": "Bengali",
            "unicode_range": (0x0980, 0x09FF),
            "halant": "\u09cd",
            "matras": [chr(c) for c in range(0x09BE, 0x09CD)],
            "tts_lang": "bn"
        },
        "ta": {
            "name": "Tamil (தமிழ்)",
            "script": "Tamil",
            "unicode_range": (0x0B80, 0x0BFF),
            "halant": "\u0bcd",
            "matras": [chr(c) for c in range(0x0BBE, 0x0BCD)],
            "tts_lang": "ta"
        },
        "te": {
            "name": "Telugu (తెలుగు)",
            "script": "Telugu",
            "unicode_range": (0x0C00, 0x0C7F),
            "halant": "\u0c4d",
            "matras": [chr(c) for c in range(0x0C3E, 0x0C4D)],
            "tts_lang": "te"
        }
    }

    # Configurable Reading Levels
    READING_LEVELS = {
        "level_1": {
            "label": "Level 1: Foundational / Low-Literacy (Grade 3-5)",
            "max_words_per_sentence": 10,
            "max_indic_complexity": 35.0,
            "style_guide": (
                "Use extremely simple everyday spoken words (Bolchal/Chalit Bhasha). "
                "Break complex ideas into short sentences (under 10 words). "
                "Explain every scientific/technical concept using rural or household analogies (farming, cooking, daily tools). "
                "Avoid conjunct consonant clusters (Samyuktakshar/Yuktakshar) wherever a simpler spoken word exists."
            )
        },
        "level_2": {
            "label": "Level 2: Intermediate / Middle School (Grade 6-8)",
            "max_words_per_sentence": 15,
            "max_indic_complexity": 55.0,
            "style_guide": (
                "Use clear, conversational educational language. Keep sentences under 15 words. "
                "Retain key English technical terms transliterated in brackets alongside their simple regional meaning."
            )
        },
        "level_3": {
            "label": "Level 3: Standard Academic / High School (Grade 9-12)",
            "max_words_per_sentence": 22,
            "max_indic_complexity": 75.0,
            "style_guide": (
                "Preserve full technical depth while eliminating unnecessary academic jargon and passive voice. "
                "Maintain standard textbook terminology paired with intuitive explanations."
            )
        }
    }

os.makedirs(Config.AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.PLOTS_OUTPUT_DIR, exist_ok=True)