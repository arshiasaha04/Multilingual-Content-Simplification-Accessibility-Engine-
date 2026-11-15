import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.config import Config


class NeuralAccessibilityAudioEngine:
    """
    Multilingual Neural Text-to-Speech Engine for Low-Literacy Accessibility.
    Runs async edge-tts inside a dedicated worker thread so it never conflicts
    with FastAPI/Uvicorn's running event loop on cloud containers.
    """

    INDIC_NEURAL_VOICES = {
        "en": "en-IN-NeerjaNeural",
        "hi": "hi-IN-SwaraNeural",
        "bn": "bn-IN-TanishaaNeural",
        "ta": "ta-IN-PallaviNeural",
        "te": "te-IN-ShrutiNeural",
        "mr": "mr-IN-AarohiNeural",
        "gu": "gu-IN-DhwaniNeural",
        "kn": "kn-IN-SapnaNeural",
        "ml": "ml-IN-SobhanaNeural",
        "ur": "ur-IN-GulNeural"
    }

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2)

    @staticmethod
    def _run_edge_tts_in_thread(text: str, voice: str, rate: str, output_path: str):
        import edge_tts

        async def _coro():
            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
            await communicate.save(output_path)

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_coro())
        finally:
            loop.close()

    def synthesize_narration(self, text: str, lang_code: str, reading_level: str) -> str:
        if not text or not text.strip():
            return ""

        filename = f"narration_{lang_code}_{uuid.uuid4().hex[:8]}.mp3"
        output_path = os.path.join(Config.AUDIO_OUTPUT_DIR, filename)

        # 1. Primary: Microsoft Edge Neural Indian Voices in an isolated event-loop thread
        if lang_code in self.INDIC_NEURAL_VOICES:
            try:
                voice = self.INDIC_NEURAL_VOICES[lang_code]
                rate = "-15%" if reading_level == "level_1" else ("-5%" if reading_level == "level_2" else "+0%")
                future = self._executor.submit(
                    self._run_edge_tts_in_thread, text, voice, rate, output_path
                )
                future.result(timeout=20)
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return f"/static/audio/{filename}"
            except Exception as e:
                print(f"[INFO] edge_tts thread fallback to gTTS for {lang_code}: {e}")

        # 2. Secondary: Google Multilingual TTS (Supports Punjabi 'pa' and all Indic scripts)
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang=lang_code, slow=(reading_level == "level_1"))
            tts.save(output_path)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return f"/static/audio/{filename}"
        except Exception as e:
            print(f"[ERROR] TTS synthesis failed for {lang_code}: {e}")

        return ""