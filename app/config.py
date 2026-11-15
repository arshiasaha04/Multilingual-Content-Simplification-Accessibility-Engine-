import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

    DATABASE_URL = "sqlite:///./accessibility_engine.db"
    AUDIO_OUTPUT_DIR = os.path.join("static", "audio")
    PLOTS_OUTPUT_DIR = os.path.join("static", "plots")

    # Supported Indian Languages & Unicode Script Ranges (All 11 Dropdown Languages)
    SUPPORTED_LANGUAGES = {
        "en": {"name": "English", "script": "Latin", "unicode_range": (0x0000, 0x007F), "halant": "", "matras": [], "tts_lang": "en"},
        "hi": {"name": "Hindi (हिन्दी)", "script": "Devanagari", "unicode_range": (0x0900, 0x097F), "halant": "\u094d", "matras": [chr(c) for c in range(0x093E, 0x094D)], "tts_lang": "hi"},
        "bn": {"name": "Bengali (বাংলা)", "script": "Bengali", "unicode_range": (0x0980, 0x09FF), "halant": "\u09cd", "matras": [chr(c) for c in range(0x09BE, 0x09CD)], "tts_lang": "bn"},
        "ta": {"name": "Tamil (தமிழ்)", "script": "Tamil", "unicode_range": (0x0B80, 0x0BFF), "halant": "\u0bcd", "matras": [chr(c) for c in range(0x0BBE, 0x0BCD)], "tts_lang": "ta"},
        "te": {"name": "Telugu (తెలుగు)", "script": "Telugu", "unicode_range": (0x0C00, 0x0C7F), "halant": "\u0c4d", "matras": [chr(c) for c in range(0x0C3E, 0x0C4D)], "tts_lang": "te"},
        "mr": {"name": "Marathi (मराठी)", "script": "Devanagari", "unicode_range": (0x0900, 0x097F), "halant": "\u094d", "matras": [chr(c) for c in range(0x093E, 0x094D)], "tts_lang": "mr"},
        "gu": {"name": "Gujarati (ગુજરાતી)", "script": "Gujarati", "unicode_range": (0x0A80, 0x0AFF), "halant": "\u0acd", "matras": [chr(c) for c in range(0x0ABE, 0x0ACD)], "tts_lang": "gu"},
        "kn": {"name": "Kannada (ಕನ್ನಡ)", "script": "Kannada", "unicode_range": (0x0C80, 0x0CFF), "halant": "\u0ccd", "matras": [chr(c) for c in range(0x0CBE, 0x0CCD)], "tts_lang": "kn"},
        "ml": {"name": "Malayalam (മലയാളം)", "script": "Malayalam", "unicode_range": (0x0D00, 0x0D7F), "halant": "\u0d4d", "matras": [chr(c) for c in range(0x0D3E, 0x0D4D)], "tts_lang": "ml"},
        "pa": {"name": "Punjabi (ਪੰਜਾਬੀ)", "script": "Gurmukhi", "unicode_range": (0x0A00, 0x0A7F), "halant": "\u0a4d", "matras": [chr(c) for c in range(0x0A3E, 0x0A4D)], "tts_lang": "pa"},
        "ur": {"name": "Urdu (اردو)", "script": "Perso-Arabic", "unicode_range": (0x0600, 0x06FF), "halant": "", "matras": [], "tts_lang": "ur"}
    }

    READING_LEVELS = {
        "level_1": {
            "label": "Level 1: Foundational / Low-Literacy (Grade 3-5)",
            "max_words_per_sentence": 10,
            "max_indic_complexity": 35.0,
            "style_guide": (
                "Use extremely simple everyday spoken words (Bolchal/Chalit Bhasha). "
                "Break complex ideas into short sentences (under 10 words). "
                "Explain every scientific/technical concept using rural or household analogies. "
                "Avoid conjunct consonant clusters wherever a simpler spoken word exists."
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
