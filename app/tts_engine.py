import os
import sys
import uuid
import subprocess
from app.config import Config


class NeuralAccessibilityAudioEngine:
    """
    Multilingual Neural Text-to-Speech Engine for Low-Literacy Accessibility.
    Uses direct in-memory UTF-8 synthesis (compatible with Windows Devanagari/Bengali)
    and adjusts speech prosody based on reading level.
    """

    INDIC_NEURAL_VOICES = {
        "hi": "hi-IN-SwaraNeural",
        "bn": "bn-IN-TanishaaNeural",
        "ta": "ta-IN-PallaviNeural",
        "te": "te-IN-ShrutiNeural"
    }

    def __init__(self):
        self._coqui_tts = None
        self._coqui_available = False
        self._ensure_tts_packages()

    def _ensure_tts_packages(self):
        """Ensures gTTS and edge-tts are installed in the exact Python environment running Uvicorn."""
        try:
            import gtts
            import edge_tts
        except ImportError:
            print("[INFO] Installing lightweight neural TTS packages into active environment...")
            subprocess.run([sys.executable, "-m", "pip", "install", "gTTS", "edge-tts"], check=False)

    def synthesize_narration(self, text: str, lang_code: str, reading_level: str) -> str:
        """
        Synthesizes natural spoken narration in Hindi, Bengali, Tamil, or Telugu
        directly in UTF-8 memory without Windows CLI encoding issues.
        """
        filename = f"narration_{lang_code}_{uuid.uuid4().hex[:8]}.mp3"
        output_path = os.path.join(Config.AUDIO_OUTPUT_DIR, filename)

        # 1. Primary: Direct in-memory Neural Edge-TTS (Supports Hindi, Bengali, Tamil, Telugu)
        try:
            import edge_tts
            voice = self.INDIC_NEURAL_VOICES.get(lang_code, "hi-IN-SwaraNeural")
            rate = "-15%" if reading_level == "level_1" else ("-5%" if reading_level == "level_2" else "+0%")
            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
            communicate.save_sync(output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return f"/static/audio/{filename}"
        except Exception as e:
            print(f"[INFO] Switching from edge_tts to gTTS ({e})")

        # 2. Direct in-memory Google Multilingual TTS (UTF-8 safe on Windows)
        try:
            from gtts import gTTS
            # Use slow=False for natural articulation
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return f"/static/audio/{filename}"
        except Exception as e:
            print(f"[ERROR] TTS generation failed: {e}")

        return ""