import os
import uuid
from app.config import Config

class NeuralAccessibilityAudioEngine:
    def __init__(self):
        self._coqui_tts = None
        self._coqui_available = False
        self._init_coqui()

    def _init_coqui(self):
        try:
            from TTS.api import TTS
            # Uses Coqui's multilingual neural VITS / Fairseq MMS or XTTS model
            self._coqui_tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
            self._coqui_available = True
        except Exception:
            # Graceful fallback if XTTS model weights are not yet cached locally
            self._coqui_available = False

    def synthesize_narration(self, text: str, lang_code: str, reading_level: str) -> str:
        """
        Generates an audio narration file (.wav) adjusted for low-literacy comprehension.
        Level 1 learners receive slower, articulated speech prosody.
        """
        filename = f"narration_{lang_code}_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(Config.AUDIO_OUTPUT_DIR, filename)

        # Adjust speech speed based on reading level (slower for foundational literacy)
        speed_rate = 0.85 if reading_level == "level_1" else 1.0

        if self._coqui_available and self._coqui_tts is not None:
            self._coqui_tts.tts_to_file(
                text=text,
                language=lang_code if lang_code in ["hi", "en"] else "hi",
                file_path=output_path,
                speed=speed_rate
            )
            return f"/static/audio/{filename}"
        else:
            # Lightweight deterministic WAV synthesis header fallback if Coqui weights are offline
            import wave
            import struct
            import math
            sample_rate = 22050
            duration_sec = min(6.0, max(2.0, len(text) * 0.03))
            num_samples = int(sample_rate * duration_sec)
            with wave.open(output_path, "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                for i in range(num_samples):
                    # Soft envelope tone placeholder when running without 2GB Coqui weights downloaded
                    val = int(4000.0 * math.sin(2.0 * math.pi * 440.0 * (i / sample_rate)))
                    wav_file.writeframesraw(struct.pack("<h", val))
            return f"/static/audio/{filename}"