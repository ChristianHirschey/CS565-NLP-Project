"""
Model 4: LLM API (Gemini)
Uses Google's Gemini API to score resume-JD matches with reasoning.
"""

import os
import json
import time
from typing import List, Tuple, Dict
from google import genai
from google.genai import types
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

class GeminiModel:
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini API client.
        
        Args:
            api_key: Gemini API key (if None, reads from GEMINI_API_KEY env variable)
        """
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
            )
        
        self.client = genai.Client(api_key=api_key)
        print("Gemini API initialized successfully!")
    
    def score_match(self, resume_text: str, jd_text: str) -> Dict[str, any]:
        """
        Score resume-JD match using Gemini LLM.
        
        Args:
            resume_text: Resume text content
            jd_text: Job description text content
            
        Returns:
            Dictionary with 'score' (float 1-10) and 'rationale' (str)
        """
        prompt = f"""You are an expert technical recruiter evaluating candidate-job fit.

Evaluate this resume against the job description on a scale of 1-10, where:
- 1-3: Poor match (missing critical requirements)
- 4-5: Moderate match (meets some requirements, gaps exist)
- 6-7: Good match (meets most requirements)
- 8-10: Excellent match (exceeds requirements)

Focus on:
1. Required skills and technologies mentioned
2. Experience level and years
3. Educational background alignment
4. Relevant project experience
5. Technical depth in required areas

Assume that a candidate is always eligible for the position, disregard any visa, location or student status issues.
However, do take years of experience into account when evaluating the match.

Scores must always be an integer or halfway between two integers (e.g., 6, 7.5).

Be strict - only give 9-10 for truly exceptional matches.

Return ONLY a JSON object with "score" (number 1-10) and "rationale" (1-2 sentences explaining the score).

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}
"""
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                )
            )
            response_text = response.text.strip()
            
            # extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # parse JSON
            result = json.loads(response_text)
            
            # validate result
            if "score" not in result or "rationale" not in result:
                raise ValueError("Response missing required keys")
            
            # ensure score is a number
            score = float(result["score"])
            if score < 1 or score > 10:
                raise ValueError(f"Score {score} out of valid range [1-10]")
            
            return {
                "score": score,
                "rationale": result["rationale"]
            }
            
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            print(f"Response: {response.text if 'response' in locals() else 'No response'}")
            # return neutral score on error
            return {
                "score": -1,
                "rationale": f"Error: {str(e)}"
            }


def run_model_4(pairs: List[Tuple[str, str]], delay: float = 15, max_pairs: int = None) -> Tuple[List[float], List[str], List[float]]:
    """
    Run Model 4 on all resume-JD pairs.
    
    Args:
        pairs: List of (resume_text, jd_text) tuples
        delay: Delay between API calls in seconds (to avoid rate limiting)
        max_pairs: Maximum number of pairs to process (None = all pairs)
        
    Returns:
        Tuple of (scores list, rationales list, inference_times list in milliseconds)
    """
    model = GeminiModel()
    model_4_scores = []
    model_4_rationales = []
    inference_times = []
    
    # limit pairs if specified
    if max_pairs:
        pairs = pairs[:max_pairs]
        print(f"WARNING: Processing only first {max_pairs} pairs to avoid quota limits")
    
    total_pairs = len(pairs)
    
    for i, (resume_text, jd_text) in enumerate(pairs, 1):
        print(f"\nProcessing Pair {i}/{total_pairs}...")
        
        start_time = time.time()
        result = model.score_match(resume_text, jd_text)
        inference_time = (time.time() - start_time) * 1000  # milliseconds
        
        score = result["score"]
        rationale = result["rationale"]
        
        model_4_scores.append(score)
        model_4_rationales.append(rationale)
        inference_times.append(inference_time)
        
        print(f"Score: {score:.2f}/10 (Time: {inference_time:.2f}ms)")
        print(f"Rationale: {rationale[:100]}...")  # print first 100 chars
        
        # avoid rate limiting
        if i < total_pairs:
            time.sleep(delay)
    
    print(f"\nAverage inference time: {sum(inference_times)/len(inference_times):.2f}ms")
    return model_4_scores, model_4_rationales, inference_times


if __name__ == "__main__":
    # example usage
    from data_loader import load_resume_jd_pairs
    
    # load 200 resume-JD pairs
    pairs = load_resume_jd_pairs()
    
    # run model 4
    print("=" * 50)
    print("MODEL 4: Gemini LLM")
    print("=" * 50)

    scores, rationales = run_model_4(pairs)
    
    print(f"\n\nSummary:")
    print(f"Average Score: {sum(scores) / len(scores):.2f}/10")
    print(f"Min Score: {min(scores):.2f}/10")
    print(f"Max Score: {max(scores):.2f}/10")
    
    # save results to file
    with open("model_4_results.json", "w") as f:
        json.dump({
            "scores": scores,
            "rationales": rationales
        }, f, indent=2)
    print("\nResults saved to model_4_results.json")
