import json
from typing import TypedDict, List, Dict, Optional
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
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=0.2
        )
        self.validator = ReadabilityAndQualityValidator()
        self.graph = self._build_graph()

    def _invoke_json_llm(self, system_prompt: str, user_prompt: str) -> dict:
        response = self.llm.invoke([
            SystemMessage(content=system_prompt + "\nAlways respond strictly in valid JSON format."),
            HumanMessage(content=user_prompt)
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())

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
Your task is to simplify complex educational content into clear, accessible English tailored for:
{level_cfg['label']}.
Rules:
1. {level_cfg['style_guide']}
2. Maximum average sentence length: {level_cfg['max_words_per_sentence']} words.
3. Do NOT omit core scientific or factual truths—explain them through concrete everyday analogies.
{feedback_instruction}

Output JSON schema:
{{
  "simplified_english": "The simplified English text..."
}}"""
        result = self._invoke_json_llm(system_prompt, f"Original Content:\n{state['original_text']}")
        simplified_en = result.get("simplified_english", state["original_text"])

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
        lang_cfg = Config.SUPPORTED_LANGUAGES[state["target_language"]]
        level_cfg = Config.READING_LEVELS[state["reading_level"]]
        feedback_instruction = ""
        if state.get("critique_feedback"):
            feedback_instruction = f"\nCorrection required from Validator: {state['critique_feedback']}"

        system_prompt = f"""You are an expert Native {lang_cfg['name']} Pedagogical Translator specializing in Accessible Education (NEP 2020).
Translate and adapt the simplified English text into {lang_cfg['name']} ({lang_cfg['script']} script) for {level_cfg['label']}.

CRITICAL ACCESSIBILITY RULES:
1. AVOID THE SANSKRITIZATION / ARCHIVAL TRAP: Do NOT use hyper-formal literary vocabulary (avoid stiff Shuddh Hindi or archaic Sadhu Bhasha). Use natural, spoken, everyday colloquial language (Bolchal ki Hindi / Chalit Bhasha) that a rural learner or low-literacy citizen easily understands.
2. TECHNICAL ANCHORING: Keep core scientific/technical nouns phonetically transliterated in {lang_cfg['script']} alongside a simple native explanation so students aren't confused by artificial textbook translations.
3. Minimize heavy conjunct consonants (Yuktaksharas) when a simpler everyday word exists.
{feedback_instruction}

Output JSON schema:
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
        result = self._invoke_json_llm(system_prompt, f"Simplified English:\n{state['simplified_english']}")
        return {
            "translated_indic_text": result.get("translated_indic_text", ""),
            "technical_glossary": result.get("technical_glossary", [])
        }

    # ---------------- NODE 3: AUTOMATED QUALITY VALIDATOR ---------------- #
    def validate_quality_node(self, state: PipelineState) -> dict:
        lang_cfg = Config.SUPPORTED_LANGUAGES[state["target_language"]]
        level_cfg = Config.READING_LEVELS[state["reading_level"]]

        # Step A: Perform Back-Translation (Indic -> English) to verify semantic retention
        bt_prompt = f"""Translate the following {lang_cfg['name']} text back into literal, accurate English without adding external knowledge.
Output JSON schema:
{{
  "back_translated_english": "Literal English back-translation..."
}}"""
        bt_res = self._invoke_json_llm(bt_prompt, state["translated_indic_text"])
        back_translated = bt_res.get("back_translated_english", "")

        # Step B: Compute Scikit-Learn Cosine Similarity & Script-Aware Indic Readability
        bt_sim = self.validator.compute_back_translation_similarity(
            state["simplified_english"], back_translated
        )
        indic_metrics = self.validator.compute_indic_readability_index(
            state["translated_indic_text"], state["target_language"]
        )
        iri_score = indic_metrics["iri_score"]

        # Step C: AI-Assisted Semantic Consistency & Hallucination Check
        audit_prompt = f"""Compare the Original Text, the Simplified {lang_cfg['name']} Translation, and its Back-Translation.
Evaluate whether core facts are preserved without hallucination and whether the vocabulary matches {level_cfg['label']}.
Output JSON schema:
{{
  "semantic_consistency_score": 0.92,
  "critique_if_failed": "Specific actionable instruction to fix overly complex words or missing facts"
}}"""
        audit_input = (
            f"Original: {state['original_text']}\n"
            f"Simplified English: {state['simplified_english']}\n"
            f"Back-Translation: {back_translated}\n"
            f"Indic Readability Score: {iri_score} (Target max: {level_cfg['max_indic_complexity']})"
        )
        audit_res = self._invoke_json_llm(audit_prompt, audit_input)
        llm_sem_score = float(audit_res.get("semantic_consistency_score", 0.85))

        # Step D: Compute Composite Quality Score & Check Thresholds
        readability_compliance = 1.0 if iri_score <= level_cfg["max_indic_complexity"] else max(
            0.4, 1.0 - ((iri_score - level_cfg["max_indic_complexity"]) / 100.0)
        )
        composite_score = round(
            (0.45 * llm_sem_score) + (0.30 * bt_sim) + (0.25 * readability_compliance), 3
        )

        passed = (
            llm_sem_score >= 0.82
            and iri_score <= (level_cfg["max_indic_complexity"] + 8.0)
        ) or (state["iteration_count"] >= state["max_iterations"])

        return {
            "back_translated_text": back_translated,
            "back_translation_sim": bt_sim,
            "indic_readability_score": iri_score,
            "indic_metrics_detail": indic_metrics,
            "llm_semantic_score": llm_sem_score,
            "composite_quality_score": composite_score,
            "validation_passed": passed,
            "critique_feedback": None if passed else audit_res.get("critique_if_failed", "Simplify sentences further and preserve all facts.")
        }

    # ---------------- CONDITIONAL EDGE ROUTING ---------------- #
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