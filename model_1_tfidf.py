"""
Model 1: Baseline TF-IDF with Cosine Similarity (Without NER)

Following Singh & Garg (2024) methodology:
Reference: "Resume Ranking With TF-IDF, Cosine Similarity and Named Entity Recognition"
- First International Conference on Data, Computation and Communication 2024

This is the baseline approach that uses:
1. TF-IDF Vectorization: Converts resumes and job descriptions into numerical vectors
   - Measures term relevance (frequency in document vs. corpus)
2. Cosine Similarity: Calculates similarity between TF-IDF vectors
   - Measures the angle between two vectors in vector space

The paper notes this approach provides a "good starting point" but has limitations:
- Cannot capture semantic meaning or context
- Relies solely on keyword matching
- May miss relevant candidates with non-standard language
- Average execution time: ~0.003 sec per resume

This baseline will be enhanced in Model 2 with NER for entity extraction.
"""

import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict


def calculate_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """
    Calculate cosine similarity between resume and job description using TF-IDF.
    
    Args:
        resume_text: Resume text content
        jd_text: Job description text content
        
    Returns:
        Cosine similarity score (0 to 1)
    """
    # create TF-IDF vectorizer with improved parameters
    vectorizer = TfidfVectorizer(
        lowercase=True,              # normalize case
        ngram_range=(1, 2),          # include bigrams (e.g., "machine learning")
        min_df=1,                    # include all terms
        sublinear_tf=True            # use log scaling for term frequency
    )
    
    # fit and transform both documents
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
    
    # calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    return float(similarity)


def run_model_1(pairs: List[Tuple[str, str]]) -> Tuple[List[float], List[float]]:
    """
    Run Model 1 on all resume-JD pairs.
    
    Args:
        pairs: List of (resume_text, jd_text) tuples
        
    Returns:
        Tuple of (scores list, inference_times list in milliseconds)
    """
    model_1_scores = []
    inference_times = []
    total_pairs = len(pairs)
    
    for i, (resume_text, jd_text) in enumerate(pairs, 1):
        start_time = time.time()
        score = calculate_tfidf_similarity(resume_text, jd_text)
        inference_time = (time.time() - start_time) * 1000  # milliseconds
        
        model_1_scores.append(score)
        inference_times.append(inference_time)
        print(f"Pair {i}/{total_pairs}: TF-IDF Score = {score:.4f} (Time: {inference_time:.2f}ms)")
    
    print(f"\nAverage inference time: {sum(inference_times)/len(inference_times):.2f}ms")
    return model_1_scores, inference_times


if __name__ == "__main__":
    # example usage
    from data_loader import load_resume_jd_pairs
    
    pairs = load_resume_jd_pairs()
    
    # run model 1
    print("=" * 50)
    print("MODEL 1: TF-IDF with Cosine Similarity")
    print("=" * 50)
    scores = run_model_1(pairs)
    
    print(f"\nAverage Score: {sum(scores) / len(scores):.4f}")
    print(f"Min Score: {min(scores):.4f}")
    print(f"Max Score: {max(scores):.4f}")
