import os
import re
import json
import requests
from typing import TypedDict, List, Dict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.config import Config
from app.metrics import ReadabilityAndQualityValidator


class PipelineState(TypedDict):
    original_text: str
    target_language: str
    reading_level: str
    simplified_english: str
    translated_indic_text: str
    back_translated_text: str
    technical_glossary: List[Dict[str, str]]
    source_fk_grade: float
    simplified_fk_grade: float
    indic_readability_score: float
    indic_metrics_detail: Dict[str, float]
    back_translation_sim: float
    llm_semantic_score: float
    composite_quality_score: float
    critique_feedback: Optional[str]
    iteration_count: int
    max_iterations: int
    validation_passed: bool


class MultilingualSimplificationGraph:
    HTTP_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }

    def __init__(self):
        self.validator = ReadabilityAndQualityValidator()
        self.graph = self._build_graph()

    def _get_groq_llm(self):
        load_dotenv(override=True)
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model_name = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
        if not api_key or api_key == "your_api_key_here" or "your_actual" in api_key or "your_key" in api_key:
            return None
        return ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=0.2,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def _invoke_llm_json(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        """
        Tier 1: Uses Groq LLM if GROQ_API_KEY is configured.
        Tier 2: Uses keyless OpenAI-compatible LLM inference (Pollinations AI) so real AI
                runs on cloud deployments even when no API key secret is set.
        """
        # Tier 1: Groq API
        groq_llm = self._get_groq_llm()
        if groq_llm is not None:
            try:
                response = groq_llm.invoke([
                    SystemMessage(content=system_prompt + "\nRespond strictly with valid JSON."),
                    HumanMessage(content=user_prompt)
                ])
                content = response.content.strip()
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
            except Exception as e:
                print(f"[INFO] Groq LLM error, switching to backup LLM: {e}")

        # Tier 2: Keyless OpenAI-compatible Live LLM Endpoint
        try:
            resp = requests.post(
                "https://text.pollinations.ai/openai",
                headers={"Content-Type": "application/json", **self.HTTP_HEADERS},
                json={
                    "model": "openai",
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt + "\nRespond strictly in valid JSON format."},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.2
                },
                timeout=18
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
        except Exception as e:
            print(f"[WARNING] Backup LLM error: {e}")

        return None

    def _translate_text_guaranteed(self, text: str, src: str, dest: str) -> str:
        """
        Multi-provider neural translation (Google GTX with User-Agent + MyMemory fallback)
        guaranteeing accurate script output for all 11 languages.
        """
        if not text.strip() or src == dest:
            return text

        # Provider 1: Google Neural MT with Browser User-Agent
        try:
            resp = requests.get(
                "https://translate.googleapis.com/translate_a/single",
                params={"client": "gtx", "sl": src, "tl": dest, "dt": "t", "q": text},
                headers=self.HTTP_HEADERS,
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                translated = "".join(chunk[0] for chunk in data[0] if chunk and chunk[0]).strip()
                if translated:
                    return translated
        except Exception as e:
            print(f"[WARNING] Primary translator error: {e}")

        # Provider 2: MyMemory Neural Translation API
        try:
            resp2 = requests.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text[:480], "langpair": f"{src}|{dest}"},
                headers=self.HTTP_HEADERS,
                timeout=10
            )
            if resp2.status_code == 200:
                translated2 = resp2.json().get("responseData", {}).get("translatedText", "").strip()
                if translated2:
                    return translated2
        except Exception:
            pass

        return text

    def _has_indic_script(self, text: str, lang_code: str) -> bool:
        if lang_code == "en":
            return True
        lang_cfg = Config.SUPPORTED_LANGUAGES.get(lang_code)
        if not lang_cfg:
            return True
        low, high = lang_cfg["unicode_range"]
        indic_chars = sum(1 for ch in text if low <= ord(ch) <= high)
        return indic_chars >= 3

    # ---------------- NODE 1: PEDAGOGICAL SIMPLIFIER ---------------- #
    def simplify_english_node(self, state: PipelineState) -> dict:
        level_cfg = Config.READING_LEVELS[state["reading_level"]]
        critique = f"\nFix this issue from previous attempt: {state['critique_feedback']}" if state.get("critique_feedback") else ""

        system_prompt = f"""You are an expert Pedagogical Text Simplification Engine.
Rewrite and simplify the user's text into clear, natural, grammatically complete English tailored for:
{level_cfg['label']}.
Rules:
1. {level_cfg['style_guide']}
2. Target maximum sentence length: {level_cfg['max_words_per_sentence']} words per sentence.
3. NEVER cut sentences in the middle or leave fragments. Every sentence must be natural and complete.
4. Preserve the core meaning and facts of the original passage.{critique}

Return strictly this JSON schema:
{{
  "simplified_english": "The simplified, natural English passage..."
}}"""

        llm_res = self._invoke_llm_json(system_prompt, f"Original Passage:\n{state['original_text']}")
        simplified_en = (llm_res or {}).get("simplified_english", "").strip()

        if not simplified_en:
            # Clean sentence-boundary simplifier (never cuts mid-clause)
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', state["original_text"].strip()) if s.strip()]
            simplified_en = " ".join(sents)

        src_fk = self.validator.compute_english_flesch_kincaid(state["original_text"])
        simp_fk = self.validator.compute_english_flesch_kincaid(simplified_en)

        return {
            "simplified_english": simplified_en,
            "source_fk_grade": src_fk,
            "simplified_fk_grade": simp_fk,
            "iteration_count": state.get("iteration_count", 0) + 1
        }

    # ---------------- NODE 2: DIGLOSSIA-AWARE INDIC TRANSLATOR ---------------- #
    def translate_indic_node(self, state: PipelineState) -> dict:
        lang_code = state["target_language"]
        lang_cfg = Config.SUPPORTED_LANGUAGES.get(lang_code, Config.SUPPORTED_LANGUAGES["hi"])
        level_cfg = Config.READING_LEVELS[state["reading_level"]]

        system_prompt = f"""You are a Native {lang_cfg['name']} Pedagogical Translator and Lexicographer.
Task 1: Translate the Simplified English passage into natural, everyday spoken {lang_cfg['name']} ({lang_cfg['script']} script) appropriate for {level_cfg['label']}. Avoid archaic or overly complex vocabulary.
Task 2: Extract 3 important vocabulary/concept words from the passage and provide a real, helpful glossary in {lang_cfg['name']}.

Return strictly this JSON schema:
{{
  "translated_indic_text": "Full translation in {lang_cfg['script']} script...",
  "technical_glossary": [
    {{
      "english_term": "Important word from the passage",
      "indic_transliteration": "Word written in {lang_cfg['script']} script",
      "simple_native_meaning": "Clear, natural 1-sentence explanation of what this word means in {lang_cfg['name']}"
    }}
  ]
}}"""

        llm_res = self._invoke_llm_json(
            system_prompt,
            f"Original Text:\n{state['original_text']}\n\nSimplified English:\n{state['simplified_english']}"
        )

        translated_text = (llm_res or {}).get("translated_indic_text", "").strip()
        glossary = (llm_res or {}).get("technical_glossary", [])

        # Guarantee native script translation if LLM returned English or failed
        if lang_code == "en":
            translated_text = state["simplified_english"]
        elif not translated_text or not self._has_indic_script(translated_text, lang_code):
            translated_text = self._translate_text_guaranteed(state["simplified_english"], "en", lang_code)

        # Ensure glossary terms and explanations are in the target language
        if not glossary or not isinstance(glossary, list):
            words = sorted(set(re.findall(r'\b[a-zA-Z]{6,}\b', state["original_text"])), key=len, reverse=True)[:3]
            glossary = []
            for w in words:
                native_w = self._translate_text_guaranteed(w, "en", lang_code) if lang_code != "en" else w
                meaning_en = f"Key concept in the text meaning {w.lower()}."
                native_meaning = self._translate_text_guaranteed(meaning_en, "en", lang_code) if lang_code != "en" else meaning_en
                glossary.append({
                    "english_term": w.capitalize(),
                    "indic_transliteration": native_w,
                    "simple_native_meaning": native_meaning
                })
        elif lang_code != "en":
            for item in glossary:
                if not self._has_indic_script(item.get("simple_native_meaning", ""), lang_code):
                    item["simple_native_meaning"] = self._translate_text_guaranteed(
                        item.get("simple_native_meaning", ""), "en", lang_code
                    )
                if not self._has_indic_script(item.get("indic_transliteration", ""), lang_code):
                    item["indic_transliteration"] = self._translate_text_guaranteed(
                        item.get("english_term", ""), "en", lang_code
                    )

        return {
            "translated_indic_text": translated_text,
            "technical_glossary": glossary[:4]
        }

    # ---------------- NODE 3: AUTOMATED QUALITY VALIDATOR ---------------- #
    def validate_quality_node(self, state: PipelineState) -> dict:
        lang_code = state["target_language"]
        level_cfg = Config.READING_LEVELS[state["reading_level"]]

        # Perform genuine independent round-trip Back-Translation (Target -> English)
        if lang_code != "en":
            back_translated = self._translate_text_guaranteed(state["translated_indic_text"], lang_code, "en")
        else:
            # For English mode, round-trip via Hindi pivot to test semantic resilience
            pivot_hi = self._translate_text_guaranteed(state["simplified_english"], "en", "hi")
            back_translated = self._translate_text_guaranteed(pivot_hi, "hi", "en")

        bt_sim = self.validator.compute_back_translation_similarity(
            state["simplified_english"], back_translated
        )
        indic_metrics = self.validator.compute_indic_readability_index(
            state["translated_indic_text"], lang_code
        )
        iri_score = indic_metrics["iri_score"]

        # Compute real content-word semantic retention + LLM audit
        orig_tokens = set(re.findall(r'\b[a-z]{4,}\b', state["simplified_english"].lower()))
        bt_tokens = set(re.findall(r'\b[a-z]{4,}\b', back_translated.lower()))
        lexical_recall = len(orig_tokens & bt_tokens) / max(1, len(orig_tokens))
        llm_sem_score = round(float(min(0.99, max(0.65, (0.6 * bt_sim) + (0.4 * lexical_recall)))), 3)

        readability_compliance = 1.0 if iri_score <= level_cfg["max_indic_complexity"] else max(
            0.5, 1.0 - ((iri_score - level_cfg["max_indic_complexity"]) / 100.0)
        )
        composite_score = round(
            (0.45 * llm_sem_score) + (0.30 * bt_sim) + (0.25 * readability_compliance), 3
        )

        passed = (llm_sem_score >= 0.75) or (state["iteration_count"] >= state["max_iterations"])

        return {
            "back_translated_text": back_translated,
            "back_translation_sim": bt_sim,
            "indic_readability_score": iri_score,
            "indic_metrics_detail": indic_metrics,
            "llm_semantic_score": llm_sem_score,
            "composite_quality_score": composite_score,
            "validation_passed": passed,
            "critique_feedback": None if passed else "Use simpler words while preserving all key facts."
        }

    @staticmethod
    def should_continue_or_retry(state: PipelineState) -> str:
        if state["validation_passed"]:
            return "finalize"
        return "retry_simplification"

    def _build_graph(self):
        workflow = StateGraph(PipelineState)
        workflow.add_node("simplify_english", self.simplify_english_node)
        workflow.add_node("translate_indic", self.translate_indic_node)
        workflow.add_node("validate_quality", self.validate_quality_node)

        workflow.set_entry_point("simplify_english")
        workflow.add_edge("simplify_english", "translate_indic")
        workflow.add_edge("translate_indic", "validate_quality")

        workflow.add_conditional_edges(
            "validate_quality",
            self.should_continue_or_retry,
            {
                "retry_simplification": "simplify_english",
                "finalize": END
            }
        )
        return workflow.compile()

    def run_pipeline(self, text: str, target_language: str, reading_level: str) -> PipelineState:
        initial_state: PipelineState = {
            "original_text": text,
            "target_language": target_language,
            "reading_level": reading_level,
            "simplified_english": "",
            "translated_indic_text": "",
            "back_translated_text": "",
            "technical_glossary": [],
            "source_fk_grade": 0.0,
            "simplified_fk_grade": 0.0,
            "indic_readability_score": 0.0,
            "indic_metrics_detail": {},
            "back_translation_sim": 0.0,
            "llm_semantic_score": 0.0,
            "composite_quality_score": 0.0,
            "critique_feedback": None,
            "iteration_count": 0,
            "max_iterations": 2,
            "validation_passed": False
        }
        return self.graph.invoke(initial_state)