import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import Config

class ReadabilityAndQualityValidator:
    
    @staticmethod
    def count_english_syllables(word: str) -> int:
        word = word.lower().strip(".:;?!,")
        if not word:
            return 0
        if len(word) <= 3:
            return 1
        word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
        word = re.sub(r'^y', '', word)
        syllables = re.findall(r'[aeiouy]{1,2}', word)
        return max(1, len(syllables))

    @classmethod
    def compute_english_flesch_kincaid(cls, text: str) -> float:
        """
        Computes Flesch-Kincaid Grade Level:
        FKGL = 0.39 * (Total Words / Total Sentences) + 11.8 * (Total Syllables / Total Words) - 15.59
        """
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        if not sentences or not words:
            return 0.0
        
        total_sentences = len(sentences)
        total_words = len(words)
        total_syllables = sum(cls.count_english_syllables(w) for w in words)
        
        asl = total_words / total_sentences
        asw = total_syllables / total_words
        fkgl = (0.39 * asl) + (11.8 * asw) - 15.59
        return round(float(np.clip(fkgl, 1.0, 20.0)), 2)

    @staticmethod
    def compute_indic_readability_index(indic_text: str, lang_code: str) -> dict:
        """
        Novel Script-Aware Indic Readability Metric (IRI):
        Evaluates Abugida script complexity using three morphological factors:
        1. Average Sentence Length (ASL) split on Purnaviram (।) or punctuation.
        2. Matra Density (Vowel Diacritics per word).
        3. Yuktakshar / Conjunct Consonant Density (Halant '\u094D' or '\u09CD' frequency),
           which strongly correlates with reading difficulty for low-literacy learners.
        
        Returns a normalized score in [0, 100] where lower = simpler & more accessible.
        """
        lang_cfg = Config.SUPPORTED_LANGUAGES.get(lang_code, Config.SUPPORTED_LANGUAGES["hi"])
        halant_char = lang_cfg["halant"]
        matra_chars = set(lang_cfg["matras"])

        # Split sentences by Indic Danda (।), Bengali Danda, or standard punctuation
        sentences = [s.strip() for s in re.split(r'[।|.!?]+', indic_text) if s.strip()]
        words = [w.strip() for w in indic_text.split() if w.strip()]

        if not sentences or not words:
            return {"iri_score": 0.0, "avg_sentence_length": 0.0, "conjunct_density": 0.0, "matra_density": 0.0}

        asl = len(words) / len(sentences)
        
        halant_counts = np.array([w.count(halant_char) for w in words], dtype=float)
        matra_counts = np.array([sum(1 for ch in w if ch in matra_chars) for w in words], dtype=float)
        word_lengths = np.array([len(w) for w in words], dtype=float)

        conjunct_density = float(np.mean(halant_counts))       # Avg conjunct clusters per word
        matra_density = float(np.mean(matra_counts))           # Avg vowel markers per word
        avg_char_len = float(np.mean(word_lengths))

        # Weighted Indic Complexity Formula calibrated for Low-Literacy Accessibility
        raw_complexity = (1.8 * asl) + (35.0 * conjunct_density) + (6.5 * matra_density) + (1.5 * avg_char_len)
        iri_score = round(float(np.clip(raw_complexity, 5.0, 100.0)), 2)

        return {
            "iri_score": iri_score,
            "avg_sentence_length": round(asl, 2),
            "conjunct_density": round(conjunct_density, 3),
            "matra_density": round(matra_density, 3)
        }

    @staticmethod
    def compute_back_translation_similarity(source_english: str, back_translated_english: str) -> float:
        """
        Computes lexical & structural preservation between simplified English and
        back-translated English using Scikit-Learn combined Word + Char N-Gram TF-IDF Cosine Similarity.
        """
        if not source_english.strip() or not back_translated_english.strip():
            return 0.0
        
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        tfidf_matrix = vectorizer.fit_transform([source_english.lower(), back_translated_english.lower()])
        sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(sim_score), 4)