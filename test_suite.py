"""
Test Suite for Resume-Job Description Matching

Evaluates 4 NLP models following established research methodologies:

Model 1: Baseline TF-IDF (Singh & Garg, 2024)
- Traditional keyword-based approach without NER
- TF-IDF vectorization + cosine similarity
- Fast (~0.003 sec/resume) but limited semantic understanding
- Reference: "Resume Ranking With TF-IDF, Cosine Similarity and Named Entity Recognition"

Model 2: Hybrid TF-IDF + NER (Singh & Garg, 2024)  
- Combines TF-IDF with Named Entity Recognition
- Extracts skills, qualifications, experience using NER
- Weighted scoring: entity relevance + full text similarity
- Performance: MAP=0.795, NDCG=0.8503, MRR=0.75 (~0.18 sec/resume)
- Reference: "Resume Ranking With TF-IDF, Cosine Similarity and Named Entity Recognition"

Model 3: BERT Semantic Similarity (Sharma et al., 2025)
- Uses Sentence-BERT for contextual embeddings
- Max sequence length: 128 tokens
- Cosine similarity on semantic vectors
- Avg similarity: 0.752, Threshold: 0.7 for matches
- Reference: "Job Description and Resume Matching System Using NLP"

Model 4: Generative LLM (Gemini API)
- LLM-based evaluation with reasoning
- Returns 1-10 score with human-readable rationale
- Tests advanced language understanding capabilities

Tests on 2 Resumes × 10 Job Descriptions (20 pairs) with manual ground truth scores.
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import spearmanr, pearsonr

# Import models
from model_1_tfidf import run_model_1
from model_2_tfidf_ner import run_model_2
from model_3_bert import run_model_3
from model_4_llm import run_model_4


class ModelEvaluator:
    """
    Evaluator for comparing model scores against manual scores.
    
    Following evaluation methodologies from:
    - Cosine similarity scoring (Sharma et al., 2025)
    - Correlation metrics (Spearman, Pearson)
    - Error metrics (MAE, MSE, RMSE)
    """
    
    def __init__(self, manual_scores: List[float]):
        """
        Initialize evaluator with manual/ground truth scores.
        
        Args:
            manual_scores: List of manual scores (20 items)
        """
        self.manual_scores = np.array(manual_scores)
        self.results = {}
    
    def evaluate_model(self, model_name: str, model_scores: List[float], 
                      score_range: tuple = (0, 1), inference_times: List[float] = None) -> Dict:
        """
        Evaluate a model's performance against manual scores.
        
        Args:
            model_name: Name of the model
            model_scores: List of model scores
            score_range: Tuple of (min, max) for model scores
            inference_times: List of inference times in milliseconds
            
        Returns:
            Dictionary with evaluation metrics
        """
        model_scores = np.array(model_scores)
        
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        # Performance metrics
        if inference_times:
            avg_time = np.mean(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            print(f"\nPerformance Metrics:")
            print(f"  Average Inference Time: {avg_time:.2f}ms")
            print(f"  Min Inference Time: {min_time:.2f}ms")
            print(f"  Max Inference Time: {max_time:.2f}ms")
        
        # Convert model scores to 1-10 scale for correlation calculation
        min_score, max_score = score_range
        if score_range == (0, 1):
            scaled_model_scores = model_scores * 9 + 1
        else:
            scaled_model_scores = model_scores
        
        # Correlation metrics (primary) - using scaled scores
        spearman_corr, spearman_p = spearmanr(self.manual_scores, scaled_model_scores)
        pearson_corr, pearson_p = pearsonr(self.manual_scores, scaled_model_scores)
        
        print(f"\nCorrelation Metrics:")
        print(f"  Spearman Correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")
        print(f"  Pearson Correlation:  {pearson_corr:.4f} (p={pearson_p:.4f})")
        
        # Store results
        results = {
            "model_name": model_name,
            "spearman_correlation": spearman_corr,
            "spearman_p_value": spearman_p,
            "pearson_correlation": pearson_corr,
            "pearson_p_value": pearson_p
        }
        
        if inference_times:
            results["avg_inference_time_ms"] = float(np.mean(inference_times))
            results["min_inference_time_ms"] = float(np.min(inference_times))
            results["max_inference_time_ms"] = float(np.max(inference_times))
        
        self.results[model_name] = results
        return results
    
    def generate_summary(self):
        """Generate and print final comparison summary."""
        print(f"\n\n{'='*60}")
        print("FINAL SUMMARY - MODEL COMPARISON")
        print(f"{'='*60}\n")
        
        # Sort by Spearman correlation
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['spearman_correlation'], 
            reverse=True
        )
        
        print("Correlation Metrics:")
        print(f"{'Rank':<6} {'Model':<20} {'Spearman':<12} {'Pearson':<12}")
        print("-" * 50)
        
        for rank, (model_name, results) in enumerate(sorted_results, 1):
            print(f"{rank:<6} {model_name:<20} "
                  f"{results['spearman_correlation']:>10.4f}  "
                  f"{results['pearson_correlation']:>10.4f}")
        
        print(f"\nBest Model (Spearman): {sorted_results[0][0]}")
        print(f"   Correlation: {sorted_results[0][1]['spearman_correlation']:.4f}")
        
        # Performance comparison
        print(f"\n\nPerformance Metrics:")
        print(f"{'Rank':<6} {'Model':<20} {'Avg Time (ms)':<15} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-" * 65)
        
        # Sort by average inference time
        models_with_times = [(name, res) for name, res in self.results.items() 
                            if 'avg_inference_time_ms' in res]
        sorted_by_time = sorted(models_with_times, 
                               key=lambda x: x[1]['avg_inference_time_ms'])
        
        for rank, (model_name, results) in enumerate(sorted_by_time, 1):
            print(f"{rank:<6} {model_name:<20} "
                  f"{results['avg_inference_time_ms']:>13.2f}  "
                  f"{results['min_inference_time_ms']:>10.2f}  "
                  f"{results['max_inference_time_ms']:>10.2f}")
        
        if sorted_by_time:
            print(f"\nFastest Model: {sorted_by_time[0][0]}")
            print(f"   Avg Time: {sorted_by_time[0][1]['avg_inference_time_ms']:.2f}ms")
        
        # Save results to JSON
        with open("evaluation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("\nResults saved to evaluation_results.json")


def load_test_pairs() -> List[Tuple[str, str]]:
    """
    Load the 20 specific resume-JD pairs for testing.
    Returns pairs in order: resume_1 × jd_1..10, resume_2 × jd_1..10
    """
    import os
    
    pairs = []
    data_dir = "data"
    
    resumes = []
    for resume_num in range(1, 21):
        resume_path = os.path.join(data_dir, "resumes", f"resume_{resume_num}.txt")
        with open(resume_path, 'r', encoding='utf-8') as f:
            resumes.append(f.read())
    
    # Load jd_1 through jd_10
    jds = []
    for jd_num in range(1, 11):
        jd_path = os.path.join(data_dir, "job_descriptions", f"jd_{jd_num}.txt")
        with open(jd_path, 'r', encoding='utf-8') as f:
            jds.append(f.read())
    
    # Create pairs: resume_1 × all JDs, then resume_2 × all JDs
    for resume_text in resumes:
        for jd_text in jds:
            pairs.append((resume_text, jd_text))
    
    return pairs


def load_manual_scores(filepath: str = "manual_scores.json") -> List[float]:
    """
    Load manual scores from manual_scores.json file.
    
    Args:
        filepath: Path to JSON file containing manual scores
        
    Returns:
        List of manual scores
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'scores' in data:
                scores_data = data['scores']
                if isinstance(scores_data, list) and len(scores_data) > 0:
                    if isinstance(scores_data[0], dict):
                        # Detailed format with pair_id, score, notes
                        return [entry['score'] for entry in scores_data]
                    else:
                        # Simple list of scores
                        return scores_data
            raise ValueError("Invalid format in test_scores.json")
    except FileNotFoundError:
        print(f"\nERROR: {filepath} not found.")
        print("Make sure test_scores.json exists in the project directory.")
        return []


def load_cached_gemini_results() -> Dict[str, List[float]]:
    """
    Load previously cached Gemini LLM results from all_model_scores.json.
    
    Returns:
        Dictionary with "scores" and "times" lists, or None if not found
    """
    try:
        with open("all_model_scores.json", "r") as f:
            data = json.load(f)
            if "Model 4 (Gemini)_scores" in data and "Model 4 (Gemini)_times_ms" in data:
                return {
                    "scores": data["Model 4 (Gemini)_scores"],
                    "times": data["Model 4 (Gemini)_times_ms"]
                }
    except FileNotFoundError:
        pass
    return None


def run_all_models(pairs: List[tuple], run_llm: bool = False) -> Dict[str, Dict[str, List[float]]]:
    """
    Run all models on the resume-JD pairs.
    
    Args:
        pairs: List of (resume_text, jd_text) tuples
        run_llm: Whether to run Model 4 (LLM) - DEFAULT IS FALSE to avoid API costs
        
    Returns:
        Dictionary mapping model names to {"scores": [...], "times": [...]}
    """
    all_results = {}
    
    # Model 1: TF-IDF
    print("\n" + "="*60)
    print("RUNNING MODEL 1: TF-IDF")
    print("="*60)
    scores, times = run_model_1(pairs)
    all_results['Model 1 (TF-IDF)'] = {"scores": scores, "times": times}
    
    # Model 2: TF-IDF + NER
    print("\n" + "="*60)
    print("RUNNING MODEL 2: TF-IDF + NER")
    print("="*60)
    scores, times = run_model_2(pairs)
    all_results['Model 2 (TF-IDF+NER)'] = {"scores": scores, "times": times}
    
    # Model 3: BERT
    print("\n" + "="*60)
    print("RUNNING MODEL 3: BERT")
    print("="*60)
    scores, times = run_model_3(pairs)
    all_results['Model 3 (BERT)'] = {"scores": scores, "times": times}
    
    # Model 4: LLM (load cached results by default)
    print("\n" + "="*60)
    print("MODEL 4: Gemini LLM")
    print("="*60)
    
    if run_llm:
        # Only run if explicitly requested
        print("WARNING: Running live Gemini API calls!")
        print("This will incur API costs and may hit rate limits.")
        print("="*60)
        try:
            model_4_scores, rationales, times = run_model_4(pairs)
            all_results['Model 4 (Gemini)'] = {"scores": model_4_scores, "times": times}
            
            # Save rationales separately
            with open("model_4_rationales.json", "w") as f:
                json.dump(rationales, f, indent=2)
        except Exception as e:
            print(f"ERROR: Model 4 failed: {e}")
            print("Attempting to load cached results instead...")
            cached = load_cached_gemini_results()
            if cached:
                all_results['Model 4 (Gemini)'] = cached
                print("Successfully loaded cached Gemini results")
    else:
        # Load cached results (default behavior)
        print("LOADING CACHED RESULTS (not running live API)")
        print("Reason: Avoiding rate limits and API costs")
        print("Note: These are results from a previous complete run")
        print("-" * 60)
        cached = load_cached_gemini_results()
        if cached:
            all_results['Model 4 (Gemini)'] = cached
            num_scores = len(cached["scores"])
            avg_time = sum(cached["times"]) / len(cached["times"])
            print(f"Loaded {num_scores} cached Gemini scores")
            print(f"  Average inference time: {avg_time:.2f}ms")
            print(f"  Total API calls saved: {num_scores}")
            print(f"  Estimated cost savings: ~${num_scores * 0.015:.2f}")
        else:
            print("WARNING: No cached Gemini results found!")
            print("Run with run_llm=True once to generate initial results")
    
    print("="*60)
    
    return all_results


def main():
    """Main testing suite execution."""
    print("="*60)
    print("RESUME-JD MATCHING: TEST SUITE")
    print("2 Resumes × 10 Job Descriptions = 20 Pairs")
    print("="*60)
    
    # Load test pairs
    print("\nLoading test data...")
    pairs = load_test_pairs()
    
    # Load manual scores
    print("\nLoading manual scores from test_scores.json...")
    manual_scores = load_manual_scores()
    
    if not manual_scores:
        print(f"\nERROR: Expected manual scores, got {len(manual_scores)}")
        return
    
    print(f"Loaded {len(manual_scores)} manual scores")
    print(f"Score range: {min(manual_scores):.1f} - {max(manual_scores):.1f}")
    
    # Run all models (run_llm=False by default to use cached Gemini results)
    print("\n" + "="*60)
    print("Running all models...")
    print("="*60)
    all_results = run_all_models(pairs, run_llm=False)  # Changed to False
    
    # Evaluate models
    print("\n\n" + "="*60)
    print("EVALUATION PHASE")
    print("="*60)
    
    evaluator = ModelEvaluator(manual_scores)
    
    # Evaluate each model
    for model_name, result_data in all_results.items():
        scores = result_data["scores"]
        times = result_data["times"]
        
        if 'Gemini' in model_name:
            evaluator.evaluate_model(model_name, scores, score_range=(1, 10), inference_times=times)
        else:
            evaluator.evaluate_model(model_name, scores, score_range=(0, 1), inference_times=times)
    
    # Generate final summary
    evaluator.generate_summary()
    
    # Save all scores and times
    save_data = {"manual_scores": manual_scores}
    for model_name, result_data in all_results.items():
        save_data[f"{model_name}_scores"] = result_data["scores"]
        save_data[f"{model_name}_times_ms"] = result_data["times"]
    
    with open("all_model_scores.json", "w") as f:
        json.dump(save_data, f, indent=2)
    
    print("\nAll model scores saved to all_model_scores.json")
    print("\nTesting suite completed!")


if __name__ == "__main__":
    main()
