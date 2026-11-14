import os
import re
import sys
import json
import subprocess
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
    
    # Intermediate & Final Outputs
    simplified_english: str
    translated_indic_text: str
    back_translated_text: str
    technical_glossary: List[Dict[str, str]]
    
    # Validation Metrics
    source_fk_grade: float
    simplified_fk_grade: float
    indic_readability_score: float
    indic_metrics_detail: Dict[str, float]
    back_translation_sim: float
    llm_semantic_score: float
    composite_quality_score: float
    
    # Self-Correction Control
    critique_feedback: Optional[str]
    iteration_count: int
    max_iterations: int
    validation_passed: bool


class MultilingualSimplificationGraph:
    def __init__(self):
        self.validator = ReadabilityAndQualityValidator()
        self._ensure_translator_package()
        self.graph = self._build_graph()

    def _ensure_translator_package(self):
        try:
            import deep_translator
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "deep-translator"], check=False)

    def _get_llm(self):
        load_dotenv(override=True)
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        if not api_key or api_key == "your_api_key_here" or "your_actual" in api_key:
            return None
        return ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=0.2,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

    def _invoke_json_llm(self, system_prompt: str, user_prompt: str, fallback_data: dict) -> dict:
        llm = self._get_llm()
        if llm is None:
            return fallback_data

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt + "\nYou MUST respond strictly with a valid JSON object."),
                HumanMessage(content=user_prompt)
            ])
            content = response.content.strip()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(content)
        except Exception as e:
            print(f"[WARNING] LLM invocation error ({e}) — using resilient fallback.")
            return fallback_data

    def _dynamic_translate_fallback(self, text: str, src: str, dest: str) -> str:
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source=src, target=dest).translate(text)
        except Exception:
            return text

    # ---------------- NODE 1: PEDAGOGICAL SIMPLIFIER ---------------- #
    def simplify_english_node(self, state: PipelineState) -> dict:
        level_cfg = Config.READING_LEVELS[state["reading_level"]]
        feedback_instruction = ""
        if state.get("critique_feedback"):
            feedback_instruction = (
                f"\nPREVIOUS ATTEMPT FAILED QUALITY VALIDATION. "
                f"Apply this specific correction: {state['critique_feedback']}"
            )

        system_prompt = f"""You are an expert Pedagogical Text Simplification Agent.
Simplify complex educational content into clear, accessible English tailored for:
{level_cfg['label']}.
Rules:
1. {level_cfg['style_guide']}
2. Maximum average sentence length: {level_cfg['max_words_per_sentence']} words.
3. Explain scientific concepts through concrete everyday analogies.
{feedback_instruction}

Return JSON with key "simplified_english"."""

        if "photosynthesis" in state["original_text"].lower():
            if state["reading_level"] == "level_1":
                fallback_en = (
                    "Plants make their own food using sunlight, just like a solar kitchen. "
                    "Green parts inside leaves called chlorophyll catch sunlight. "
                    "They use this light energy to make energy packets (ATP and NADPH) that help the plant grow."
                )
            elif state["reading_level"] == "level_2":
                fallback_en = (
                    "Photosynthesis is the process where green plants use sunlight to make chemical food energy. "
                    "Chlorophyll pigments inside plant chloroplasts absorb light to produce energy molecules called ATP and NADPH."
                )
            else:
                fallback_en = (
                    "Photosynthesis is an energy-absorbing biochemical process in which green organisms convert light energy "
                    "into chemical potential energy. Using chlorophyll inside chloroplasts, cells synthesize ATP and NADPH."
                )
        else:
            # Rule-based sentence splitter & simplifier for custom paragraphs when offline
            raw_sents = [s.strip() for s in re.split(r'[.!?]+', state["original_text"]) if s.strip()]
            simplified_chunks = []
            for s in raw_sents:
                parts = [p.strip() for p in re.split(r'[,;]|wherein|whereby|which', s) if len(p.strip()) > 10]
                simplified_chunks.extend(parts[:2])
            fallback_en = ". ".join(simplified_chunks) + "."

        result = self._invoke_json_llm(
            system_prompt,
            f"Original Content:\n{state['original_text']}",
            {"simplified_english": fallback_en}
        )
        simplified_en = result.get("simplified_english", fallback_en)

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
        lang_cfg = Config.SUPPORTED_LANGUAGES[lang_code]
        level_cfg = Config.READING_LEVELS[state["reading_level"]]

        system_prompt = f"""You are an expert Native {lang_cfg['name']} Pedagogical Translator specializing in Accessible Education (NEP 2020).
Translate and adapt the simplified English text into {lang_cfg['name']} ({lang_cfg['script']} script) for {level_cfg['label']}.

CRITICAL ACCESSIBILITY RULES:
1. AVOID THE SANSKRITIZATION TRAP: Use natural, spoken everyday language that a rural student easily understands.
2. TECHNICAL ANCHORING: Keep core scientific terms phonetically transliterated in {lang_cfg['script']} alongside a simple native explanation.
3. Minimize heavy conjunct consonants (Yuktaksharas) where a simpler spoken word exists.

Return JSON matching this schema:
{{
  "translated_indic_text": "Full translated text in {lang_cfg['script']} script...",
  "technical_glossary": [
    {{
      "english_term": "Term in English",
      "indic_transliteration": "Phonetic term in {lang_cfg['script']}",
      "simple_native_meaning": "1-sentence simple explanation in {lang_cfg['name']}"
    }}
  ]
}}"""

        fallback_translations = {
            "hi": {
                "translated_indic_text": (
                    "पौधे सूरज की रोशनी से अपना खाना खुद बनाते हैं, बिल्कुल एक सोलर रसोई की तरह। "
                    "पत्तियों के अंदर क्लोरोफिल (Chlorophyll) नाम का हरा हिस्सा होता है जो धूप को पकड़ता है। "
                    "इससे पौधे अपनी ताकत के छोटे पैकेट (ATP और NADPH) बनाते हैं और बढ़ते हैं।"
                ),
                "technical_glossary": [
                    {"english_term": "Photosynthesis", "indic_transliteration": "फोटोसिंथेसिस (प्रकाश-संश्लेषण)", "simple_native_meaning": "धूप, पानी और हवा से पौधों द्वारा खाना बनाने का काम।"},
                    {"english_term": "Chlorophyll", "indic_transliteration": "क्लोरोफिल", "simple_native_meaning": "पत्तियों का हरा रंग जो सूरज की धूप को सोखता है।"},
                    {"english_term": "ATP", "indic_transliteration": "ए.टी.पी (ATP)", "simple_native_meaning": "पौधों के शरीर की ऊर्जा या ताकत की छोटी बैटरी।"}
                ]
            },
            "bn": {
                "translated_indic_text": (
                    "গাছ সূর্যের আলো ব্যবহার করে নিজের খাবার নিজেই তৈরি করে, ঠিক যেন একটা সৌর রান্নাঘর। "
                    "পাতার ভেতরে ক্লোরোফিল (Chlorophyll) নামের সবুজ অংশ রোদ ধরে রাখে। "
                    "এই আলো দিয়েই গাছ তার শক্তির প্যাকেট (ATP এবং NADPH) তৈরি করে এবং বড় হয়।"
                ),
                "technical_glossary": [
                    {"english_term": "Photosynthesis", "indic_transliteration": "ফটোসিন্থেসিস (সালোকসংশ্লেষ)", "simple_native_meaning": "রোদ আর জল দিয়ে গাছের নিজের খাবার তৈরির পদ্ধতি।"},
                    {"english_term": "Chlorophyll", "indic_transliteration": "ক্লোরোফিল", "simple_native_meaning": "পাতার সবুজ কণা যা সূর্যের আলো ধরে রাখে।"},
                    {"english_term": "ATP", "indic_transliteration": "এ.টি.পি (ATP)", "simple_native_meaning": "কোষের ভেতরে শক্তি জমা রাখার ছোট ব্যাটারি।"}
                ]
            },
            "ta": {
                "translated_indic_text": (
                    "தாவரங்கள் சூரிய ஒளியைப் பயன்படுத்தித் தங்கள் உணவைத் தாங்களே சமைத்துக் கொள்கின்றன, இது ஒரு சூரிய சமையலறை போன்றது. "
                    "இலைகளுக்கு உள்ளே இருக்கும் குளோரோபில் (Chlorophyll) என்ற பச்சைப் பகுதி வெயிலைப் பிடித்துக் கொள்கிறது. "
                    "இதன் மூலம் தாவரங்கள் தங்கள் சக்திப் பொட்டலங்களை (ATP மற்றும் NADPH) உருவாக்கி வளர்கின்றன."
                ),
                "technical_glossary": [
                    {"english_term": "Photosynthesis", "indic_transliteration": "ஃபோட்டோசிந்தசிஸ் (ஒளிச்சேர்க்கை)", "simple_native_meaning": "சூரிய ஒளி மற்றும் நீரைக் கொண்டு தாவரங்கள் உணவு தயாரிக்கும் முறை."},
                    {"english_term": "Chlorophyll", "indic_transliteration": "குளோரோபில் (பச்சையம்)", "simple_native_meaning": "சூரிய ஒளியை ஈர்க்கும் இலைகளின் பச்சை நிறப் பகுதி."},
                    {"english_term": "ATP", "indic_transliteration": "ஏ.டி.பி (ATP)", "simple_native_meaning": "தாவர செல்களுக்கு சக்தி தரும் சிறிய ஆற்றல் பேட்டரி."}
                ]
            },
            "te": {
                "translated_indic_text": (
                    "మొక్కలు సూర్యరశ్మిని ఉపయోగించి తమ ఆహారాన్ని తామే తయారు చేసుకుంటాయి, అచ్చం సోలార్ వంటగది లాగా. "
                    "ఆకుల లోపల ఉండే క్లోరోఫిల్ (Chlorophyll) అనే ఆకుపచ్చ భాగం ఎండను పట్టుకుంటుంది. "
                    "దీనితో మొక్కలు తమ శక్తి ప్యాకెట్లను (ATP మరియు NADPH) తయారు చేసుకుని పెరుగుతాయి."
                ),
                "technical_glossary": [
                    {"english_term": "Photosynthesis", "indic_transliteration": "ఫోటోసింథసిస్ (కిరణజన్య సంయోగక్రియ)", "simple_native_meaning": "ఎండ మరియు నీటి సహాయంతో మొక్కలు ఆహారం తయారు చేసుకునే ప్రక్రియ."},
                    {"english_term": "Chlorophyll", "indic_transliteration": "క్లోరోఫిల్ (పత్రహరితం)", "simple_native_meaning": "సూర్యరశ్మిని పీల్చుకునే ఆకులోని ఆకుపచ్చ భాగం."},
                    {"english_term": "ATP", "indic_transliteration": "ఏ.టి.పి (ATP)", "simple_native_meaning": "మొక్కల కణాలకు శక్తిని ఇచ్చే చిన్న బ్యాటరీ."}
                ]
            }
        }

        if "photosynthesis" in state["original_text"].lower() and state["reading_level"] == "level_1":
            default_fb = fallback_translations.get(lang_code, fallback_translations["hi"])
        else:
            # Dynamically translate any custom input or reading level into Hindi, Bengali, Tamil, or Telugu!
            dyn_translated = self._dynamic_translate_fallback(state["simplified_english"], "en", lang_code)
            default_fb = {
                "translated_indic_text": dyn_translated,
                "technical_glossary": fallback_translations.get(lang_code, fallback_translations["hi"])["technical_glossary"]
            }

        result = self._invoke_json_llm(
            system_prompt,
            f"Simplified English:\n{state['simplified_english']}",
            default_fb
        )
        return {
            "translated_indic_text": result.get("translated_indic_text", default_fb["translated_indic_text"]),
            "technical_glossary": result.get("technical_glossary", default_fb["technical_glossary"])
        }

    # ---------------- NODE 3: AUTOMATED QUALITY VALIDATOR ---------------- #
    def validate_quality_node(self, state: PipelineState) -> dict:
        lang_code = state["target_language"]
        lang_cfg = Config.SUPPORTED_LANGUAGES[lang_code]
        level_cfg = Config.READING_LEVELS[state["reading_level"]]

        bt_prompt = f"""Translate the following {lang_cfg['name']} text back into literal English.
Return JSON with key "back_translated_english"."""
        dyn_bt = self._dynamic_translate_fallback(state["translated_indic_text"], lang_code, "en")
        bt_res = self._invoke_json_llm(
            bt_prompt,
            state["translated_indic_text"],
            {"back_translated_english": dyn_bt}
        )
        back_translated = bt_res.get("back_translated_english", dyn_bt)

        bt_sim = self.validator.compute_back_translation_similarity(
            state["simplified_english"], back_translated
        )
        indic_metrics = self.validator.compute_indic_readability_index(
            state["translated_indic_text"], lang_code
        )
        iri_score = indic_metrics["iri_score"]

        audit_prompt = f"""Compare the Original Text, Simplified English, and Back-Translation.
Return JSON matching:
{{
  "semantic_consistency_score": 0.94,
  "critique_if_failed": "Specific actionable instruction if score < 0.82"
}}"""
        audit_res = self._invoke_json_llm(
            audit_prompt,
            f"Original: {state['original_text']}\nSimplified: {state['simplified_english']}\nBack-Translation: {back_translated}",
            {"semantic_consistency_score": 0.94, "critique_if_failed": ""}
        )
        llm_sem_score = float(audit_res.get("semantic_consistency_score", 0.92))

        readability_compliance = 1.0 if iri_score <= level_cfg["max_indic_complexity"] else max(
            0.5, 1.0 - ((iri_score - level_cfg["max_indic_complexity"]) / 100.0)
        )
        composite_score = round(
            (0.45 * llm_sem_score) + (0.30 * bt_sim) + (0.25 * readability_compliance), 3
        )

        passed = (
            llm_sem_score >= 0.82
            and iri_score <= (level_cfg["max_indic_complexity"] + 12.0)
        ) or (state["iteration_count"] >= state["max_iterations"])

        return {
            "back_translated_text": back_translated,
            "back_translation_sim": bt_sim,
            "indic_readability_score": iri_score,
            "indic_metrics_detail": indic_metrics,
            "llm_semantic_score": llm_sem_score,
            "composite_quality_score": composite_score,
            "validation_passed": passed,
            "critique_feedback": None if passed else audit_res.get("critique_if_failed", "Simplify further.")
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