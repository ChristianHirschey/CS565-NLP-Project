# Resume-JD Matching: Quick Setup Guide

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python test_suite.py
```

**Runtime:** ~2-3 minutes | **Output:** Console results + evaluation_results.json

## Project Overview

Evaluates 4 NLP models on 200 resume-job description pairs (20 resumes × 10 JDs):

- **Model 1:** TF-IDF baseline (keyword matching)
- **Model 2:** TF-IDF + NER (hybrid with entity extraction)
- **Model 3:** BERT cross-encoder (semantic similarity)
- **Model 4:** Gemini LLM (cached results)

## Required Files

### Code Files (7 files)

- `test_suite.py` - Main evaluation script
- `model_1_tfidf.py` - TF-IDF implementation
- `model_2_tfidf_ner.py` - TF-IDF + NER implementation
- `model_3_bert.py` - BERT implementation
- `model_4_llm.py` - Gemini wrapper (not executed)
- `data_loader.py` - Data loading utilities
- `requirements.txt` - Python dependencies

### Data Files (3 items)

- `manual_scores.json` - 200 human ground truth scores
- `all_model_scores.json` - Cached Gemini results (avoids $3 API cost)
- `data/` folder:
  - `resumes/` - 20 resume text files (resume_1.txt to resume_20.txt)
  - `job_descriptions/` - 10 JD files (jd_1.txt to jd_10.txt)

---

## Installation

**Prerequisites:** Python 3.8+ with pip

1. **Install dependencies** (~2-5 min):
   ```bash
   pip install -r requirements.txt
   ```

2. **Download spaCy model** (~30 sec):
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Run test suite:**
   ```bash
   python test_suite.py
   ```

## What Happens When You Run It

1. Loads 200 resume-JD pairs from `data/`
2. Loads 200 manual scores from `manual_scores.json`
3. Runs Models 1-3 (TF-IDF, NER, BERT)
4. Loads cached Model 4 results (no API calls)
5. Evaluates all models with statistical metrics
6. Saves results to `evaluation_results.json`

**Model 4 Note:** Uses pre-computed Gemini results to avoid API rate limits and costs.

---

## Testing with Your Own Data

### Step 1: Add Resume Files
1. Create plain text files in `data/resumes/`
2. Name them: `resume_1.txt`, `resume_2.txt`, etc.
3. Format: Plain text, no special formatting needed

### Step 2: Add Job Description Files
1. Create plain text files in `data/job_descriptions/`
2. Name them: `jd_1.txt`, `jd_2.txt`, etc.
3. Format: Plain text job requirements/descriptions

### Step 3: Update Test Pairs (Optional)
If you want to evaluate on your new data:

**Option A: Quick Test (No Manual Scoring)**
```python
# In test_suite.py, modify load_test_pairs() to load your files
# Example: Load resumes 1-10 against JDs 1-10
for resume_num in range(1, 11):  # Add your resume numbers
    resume_path = os.path.join(data_dir, "resumes", f"resume_{resume_num}.txt")
    # ... rest of code
```

**Option B: Full Evaluation (With Manual Scoring)**
1. Add scores to `manual_scores.json`:
```json
{
  "scores": [
    {
      "pair_id": 1,
      "resume_file": "resume_1.txt",
      "jd_file": "jd_1.txt",
      "score": 7.5
    },
    {
      "pair_id": 2,
      "resume_file": "resume_1.txt",
      "jd_file": "jd_2.txt",
      "score": 4.0
    }
  ]
}
```
2. Update `load_test_pairs()` to include your files
3. Run `python test_suite.py`


---

## Expected Results

| Model      | Spearman | Pearson | P-value | Time/Resume | Best For          |
|------------|----------|---------|---------|-------------|-------------------|
| Gemini     | 0.65     | 0.59    | 2e-25   | 911ms       | Final evaluation  |
| TF-IDF+NER | 0.43     | 0.47    | 2e-10   | 470ms       | Primary screening |
| TF-IDF     | 0.38     | 0.42    | 2e-08   | 4ms         | Mass filtering    |
| BERT       | 0.30     | 0.20    | 2e-05   | 73ms        | N/A               |

**Interpretation:** Higher correlation = better ranking. All p-values << 0.05 = statistically significant.

## Output Files

- `evaluation_results.json` - Complete statistical results for all models
- `all_model_scores.json` - Raw scores for each of 200 pairs (updated)
- Console output - Detailed evaluation report with tables

## File Descriptions

| File                      | Purpose                                                      |
|---------------------------|--------------------------------------------------------------|
| `test_suite.py`           | Main script - loads data, runs models, evaluates performance |
| `model_1_tfidf.py`        | TF-IDF vectorization + cosine similarity (Singh & Garg 2024) |
| `model_2_tfidf_ner.py`    | TF-IDF + spaCy NER for entity extraction (Singh & Garg 2024) |
| `model_3_bert.py`         | BERT cross-encoder for semantic ranking (Sharma et al. 2025) |
| `model_4_llm.py`          | Gemini API wrapper (not called - uses cached results)        |
| `data_loader.py`          | Utility functions to load resumes/JDs from filesystem        |
| `evaluation_results.json` | Model correlation scores and performance metrics             |