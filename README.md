# Multilingual Content Simplification & Accessibility Engine

A huge chunk of India cannot access English government, educational, and health content. Even when Hindi or Bengali versions exist, they are often too complex for low-literacy readers.  
This project addresses that gap with a multilingual simplification and accessibility pipeline.

## Problem It Solves

This engine converts complex English informational content into simpler, more accessible regional-language versions, with quality checks and audio narration for low-literacy users.

## Resume Bullet (ATS-Optimized)

Built a generative pipeline that translates and simplifies English educational/informational content into regional Indian languages at controllable reading levels, with automated fidelity validation (back-translation scoring) and audio narration for low-literacy accessibility.

**Keywords:** LLM fine-tuning/prompting, multilingual NLP, content generation, evaluation pipelines, TTS, accessibility.

## Tech Stack

- **Translation & Simplification:** LLM API with carefully engineered prompts, or a fine-tuned small open model (IndicBART/IndicTrans) for cost efficiency
- **Evaluation:** Back-translation + BLEU/BERTScore for fidelity; readability formula adapted per target language for simplicity scoring
- **Text-to-Speech:** Coqui TTS or an Indic TTS model for audio output
- **Backend:** FastAPI
- **Frontend:** Simple web interface for input/output

## Build Approach

1. **Ingest source content**  
   Take English government notices, health advisories, or textbook chapters.

2. **Simplification agent**  
   Rewrite the text into simpler English first (controllable grade level via prompt), then translate it to the target language.

3. **Fidelity check**  
   Back-translate the simplified target-language text to English and compare semantic similarity with the original.  
   This quantified validation step differentiates a production-grade system from a demo.

4. **Readability scoring**  
   Score output against a target reading level and iterate/regenerate when thresholds are not met.

5. **Audio accessibility layer**  
   Convert validated final text into speech so low-literacy users can listen instead of read.

6. **Demo packaging**  
   User pastes a government circular and receives:
   - simplified Hindi/Bengali text output
   - corresponding audio narration

## Outcome

An end-to-end, socially impactful multilingual NLP application focused on accessibility, trustworthiness, and usability for underserved populations.
