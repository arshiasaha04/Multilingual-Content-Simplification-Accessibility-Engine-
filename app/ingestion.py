import os
import tempfile
from playwright.async_api import async_playwright
import whisper

class ContentIngestionEngine:
    def __init__(self):
        self._whisper_model = None

    @property
    def whisper_model(self):
        if self._whisper_model is None:
            # Uses lightweight 'base' model for fast local CPU/GPU transcription
            self._whisper_model = whisper.load_model("base")
        return self._whisper_model

    async def extract_from_url(self, url: str) -> str:
        """
        Uses Playwright headless Chromium to render JavaScript-heavy educational portals
        (e.g., NCERT, Wikipedia, SWAYAM, Khan Academy) and extract clean article body text.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            
            # Remove navigation, ads, footers, and scripts before extracting educational paragraphs
            extracted_text = await page.evaluate("""() => {
                const selectorsToRemove = ['nav', 'header', 'footer', 'script', 'style', 'aside', '.ads'];
                selectorsToRemove.forEach(sel => {
                    document.querySelectorAll(sel).forEach(el => el.remove());
                });
                const paragraphs = Array.from(document.querySelectorAll('article p, main p, p'));
                return paragraphs
                    .map(p => p.innerText.trim())
                    .filter(text => text.length > 40)
                    .slice(0, 15)
                    .join('\\n\\n');
            }""")
            await browser.close()
            
            if not extracted_text:
                raise ValueError("Could not extract readable educational paragraphs from the provided URL.")
            return extracted_text

    def transcribe_audio_lecture(self, file_bytes: bytes, filename: str) -> str:
        """
        Uses OpenAI Whisper to transcribe spoken educational lectures into text for simplification.
        """
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            result = self.whisper_model.transcribe(tmp_path, fp16=False)
            return result.get("text", "").strip()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)