# Resume-JD Matching: Quick Setup Guide

## Quick Start
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python test_suite.py

Runtime: ~2-3 minutes | Output: Console results + evaluation_results.json

## Project Overview
Evaluates 4 NLP models on 200 resume-job description pairs (20 resumes × 10 JDs):
- Model 1: TF-IDF baseline (keyword matching)
- Model 2: TF-IDF + NER (hybrid with entity extraction)
- Model 3: BERT cross-encoder (semantic similarity)
- Model 4: Gemini LLM (cached results)

## Required Files

### Code Files (7 files)
- test_suite.py - Main evaluation script
- model_1_tfidf.py - TF-IDF implementation
- model_2_tfidf_ner.py - TF-IDF + NER implementation
- model_3_bert.py - BERT implementation
- model_4_llm.py - Gemini wrapper (not executed)
- data_loader.py - Data loading utilities
- requirements.txt - Python dependencies

### Data Files (3 items)
- manual_scores.json - 200 human ground truth scores
- all_model_scores.json - Cached Gemini results (avoids $3 API cost)
- data/ folder:
  - resumes/ - 20 resume text files (resume_1.txt to resume_20.txt)
  - job_descriptions/ - 10 JD files (jd_1.txt to jd_10.txt)

---

## Installation

Prerequisites: Python 3.8+ with pip

1. Install dependencies (~2-5 min):
   pip install -r requirements.txt

2. Download spaCy model (~30 sec):
   python -m spacy download en_core_web_sm

3. Run test suite:
   python test_suite.py

## What Happens When You Run It

1. Loads 200 resume-JD pairs from data/
2. Loads 200 manual scores from manual_scores.json
3. Runs Models 1-3 (TF-IDF, NER, BERT)
4. Loads cached Model 4 results (no API calls)
5. Evaluates all models with statistical metrics
6. Saves results to evaluation_results.json

Model 4 Note: Uses pre-computed Gemini results to avoid API rate limits and costs.

## Expected Results

|     Model      | Spearman | Pearson | P-value | Time/Resume |      Best For     |
|----------------|----------|---------|---------|-------------|-------------------|
|     Gemini     |   0.65   |   0.59  |  2e-25  |    911ms    | Final evaluation  |
|   TF-IDF+NER   |   0.43   |   0.47  |  2e-10  |    470ms    | Primary screening |
|     TF-IDF     |   0.38   |   0.42  |  2e-08  |     4ms     |   Mass filtering  |
|      BERT      |   0.30   |   0.20  |  2e-05  |     73ms    |        N/A        |

Interpretation: Higher correlation = better ranking. All p-values << 0.05 = statistically significant.

## Output Files

- evaluation_results.json - Complete statistical results for all models
- all_model_scores.json - Raw scores for each of 200 pairs (updated)
- Console output - Detailed evaluation report with tables

## File Descriptions

| File                  | Purpose                                                      |
|-----------------------|--------------------------------------------------------------|
| test_suite.py         | Main script - loads data, runs models, evaluates performance |
| model_1_tfidf.py      | TF-IDF vectorization + cosine similarity (Singh & Garg 2024) |
| model_2_tfidf_ner.py  | TF-IDF + spaCy NER for entity extraction (Singh & Garg 2024) |
| model_3_bert.py       | BERT cross-encoder for semantic ranking (Sharma et al. 2025) |
| model_4_llm.py        | Gemini API wrapper (not called - uses cached results)        |
| data_loader.py        | Utility functions to load resumes/JDs from filesystem        |
| manual_scores.json    | Human-annotated ground truth (1-10 scale)                    |
| all_model_scores.json | Cached Gemini scores + raw outputs from all models           |

## Project Structure
Project/
├── test_suite.py
├── model_1_tfidf.py
├── model_2_tfidf_ner.py
├── model_3_bert.py
├── model_4_llm.py
├── data_loader.py
├── requirements.txt
├── manual_scores.json
├── all_model_scores.json
└── data/
    ├── resumes/             (20 files)
    └── job_descriptions/    (10 files)