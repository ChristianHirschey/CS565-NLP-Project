"""
Model 3: BERT Semantic Similarity
Uses Sentence-BERT cross-encoder for resume-JD matching (ranking-optimized).

Based on: Sharma et al. (2025) - "Job Description and Resume Matching System Using NLP"

Key specifications from paper:
- BERT model: bert-base-uncased with fine-tuning
- Sequence length: Max 128 tokens
- Similarity metric: Cosine similarity between embeddings
- Average similarity: 0.752
- Threshold for matches: 0.7

We use Cross-Encoder instead of Bi-Encoder because:
- Cross-encoders are specifically trained for ranking/matching tasks
- They process both texts together (not separately) for better interaction
- Better for determining "does A match B" vs just "are A and B similar"
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple
import time


class BERTCrossEncoderModel:
    """
    BERT-based cross-encoder model for resume-JD matching.
    Cross-encoders are better for ranking than bi-encoders.
    """
    
    def __init__(self):
        """Initialize the BERT cross-encoder model."""
        # use cross-encoder trained on MS MARCO passage ranking
        # this is trained to rank text passages, perfect for resume matching
        print("Loading BERT Cross-Encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Model loaded successfully")
    
    def calculate_match_score(self, resume_text: str, jd_text: str) -> float:
        """
        Calculate match score between resume and job description.
        Cross-encoder processes both texts together for ranking.
        
        Args:
            resume_text: Resume content
            jd_text: Job description content
            
        Returns:
            Match score (0-1 scale after sigmoid normalization)
        """
        # cross-encoder returns a score (not bounded 0-1)
        # the score represents how well resume matches JD
        score = self.model.predict([(jd_text, resume_text)])[0]
        
        # apply sigmoid to normalize to 0-1 range
        # this makes scores comparable across pairs
        import numpy as np
        normalized_score = 1 / (1 + np.exp(-score))
        
        return float(normalized_score)


def run_model_3(pairs: List[Tuple[str, str]]) -> Tuple[List[float], List[float]]:
    """
    Run BERT cross-encoder model on all resume-JD pairs.
    
    Args:
        pairs: List of (resume_text, jd_text) tuples
        
    Returns:
        Tuple of (scores list, inference_times list in milliseconds)
    """
    print(f"Initializing BERT Cross-Encoder model...")
    model = BERTCrossEncoderModel()
    
    model_3_scores = []
    inference_times = []
    total_pairs = len(pairs)
    
    print(f"\nProcessing {total_pairs} resume-JD pairs...")
    
    for i, (resume_text, jd_text) in enumerate(pairs, 1):
        start_time = time.time()
        
        # calculate match score
        score = model.calculate_match_score(resume_text, jd_text)
        
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        
        model_3_scores.append(score)
        inference_times.append(inference_time_ms)
        
        # print progress
        print(f"Pair {i}/{total_pairs}: BERT Cross-Encoder Score = {score:.4f} "
              f"(Time: {inference_time_ms:.2f}ms)")
    
    avg_score = sum(model_3_scores) / len(model_3_scores)
    avg_time = sum(inference_times) / len(inference_times)
    
    print(f"\n{'='*60}")
    print(f"Model 3 Complete:")
    print(f"  Average match score: {avg_score:.4f}")
    print(f"  Average inference time: {avg_time:.2f}ms")
    print(f"{'='*60}")
    
    return model_3_scores, inference_times


if __name__ == "__main__":
    # example usage
    from data_loader import load_resume_jd_pairs
    
    print("="*60)
    print("MODEL 3: BERT Cross-Encoder Test")
    print("="*60)
    
    pairs = load_resume_jd_pairs()
    print(f"\nTesting on {len(pairs)} pairs...")
    
    # run model 3
    scores, times = run_model_3(pairs)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Scores: {[f'{s:.4f}' for s in scores[:5]]}...")
    print(f"Inference times: {[f'{t:.2f}ms' for t in times[:5]]}...")