"""
Data Loader Utility
Helper module to load resume-JD pairs from your data directory.
"""

import os
import json
from typing import List, Tuple


def load_resume_jd_pairs(data_dir: str = "data") -> List[Tuple[str, str]]:
    """
    Load ALL combinations of resumes and job descriptions.
    Creates pairs of every resume with every job description.
    
    Expected directory structure:
    data/
        resumes/
            resume_1.txt
            resume_2.txt
            ...
        job_descriptions/
            jd_1.txt
            jd_2.txt
            ...
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        List of (resume_text, jd_text) tuples for all combinations
    """
    pairs = []
    
    resume_dir = os.path.join(data_dir, "resumes")
    jd_dir = os.path.join(data_dir, "job_descriptions")
    
    # Check if directories exist
    if not os.path.exists(resume_dir):
        print(f"Warning: {resume_dir} not found. Creating example data...")
        create_example_data(data_dir)
    
    # Get sorted list of files
    unsorted_resumes_1 = [f for f in os.listdir(resume_dir) if f.endswith('.txt') and len(f) <= 12]
    unsorted_resumes_2 = [f for f in os.listdir(resume_dir) if f.endswith('.txt') and len(f) >= 13]

    resume_files = sorted(unsorted_resumes_1) + sorted(unsorted_resumes_2)
    print(resume_files)

    unsorted_jds_1 = [f for f in os.listdir(jd_dir) if f.endswith('.txt') and len(f) <= 8]
    unsorted_jds_2 = [f for f in os.listdir(jd_dir) if f.endswith('.txt') and len(f) >= 9]

    jd_files = sorted(unsorted_jds_1) + sorted(unsorted_jds_2)
    print(jd_files)
    
    # Load all resumes
    resumes = []
    for resume_file in resume_files:
        resume_path = os.path.join(resume_dir, resume_file)
        with open(resume_path, 'r', encoding='utf-8') as f:
            resumes.append((resume_file, f.read()))
    
    # Load all job descriptions
    jds = []
    for jd_file in jd_files:
        jd_path = os.path.join(jd_dir, jd_file)
        with open(jd_path, 'r', encoding='utf-8') as f:
            jds.append((jd_file, f.read()))
    
    # Create all combinations: every resume with every JD
    for resume_file, resume_text in resumes:
        for jd_file, jd_text in jds:
            pairs.append((resume_text, jd_text))
    
    print(f"Loaded {len(resumes)} resumes and {len(jds)} job descriptions")
    print(f"Created {len(pairs)} total pairs (all combinations)")
    return pairs


def create_example_data(data_dir: str = "data"):
    """
    Create example data structure with placeholder files.
    This is a helper function for testing.
    """
    os.makedirs(os.path.join(data_dir, "resumes"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "job_descriptions"), exist_ok=True)
    
    # Create example resume
    example_resume = """John Doe
    
Skills
Python, Machine Learning, Data Analysis, SQL, TensorFlow, PyTorch

Experience
Senior Data Scientist at Tech Corp (2020-2023)
- Developed machine learning models for customer segmentation
- Implemented NLP solutions for text classification
- Led team of 3 data scientists

Education
MS in Computer Science, Stanford University
"""
    
    # Create example job description
    example_jd = """Data Scientist Position

We are seeking an experienced Data Scientist to join our team.

Requirements:
- 3+ years of experience in machine learning
- Strong Python programming skills
- Experience with TensorFlow or PyTorch
- Knowledge of NLP techniques
- SQL proficiency

Responsibilities:
- Build and deploy ML models
- Analyze large datasets
- Collaborate with engineering teams
"""
    
    # Write example files
    with open(os.path.join(data_dir, "resumes", "resume_1.txt"), 'w') as f:
        f.write(example_resume)
    
    with open(os.path.join(data_dir, "job_descriptions", "jd_1.txt"), 'w') as f:
        f.write(example_jd)
    
    print(f"Created example data in {data_dir}/")
    print("Please replace with your actual 20 resume-JD pairs.")


def load_manual_scores(scores_file: str = "manual_scores.json") -> List[float]:
    """
    Load manual scores from JSON file.
    
    Args:
        scores_file: Path to the manual scores JSON file
        
    Returns:
        List of manual scores (1-10 scale)
    """
    if not os.path.exists(scores_file):
        print(f"Warning: {scores_file} not found.")
        print("Run 'python generate_manual_scores_template.py' to create a template.")
        return []
    
    with open(scores_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract scores from the JSON structure
    scores = [entry["score"] for entry in data["scores"]]
    
    print(f"Loaded {len(scores)} manual scores from {scores_file}")
    return scores


if __name__ == "__main__":
    # Test the data loader
    pairs = load_resume_jd_pairs()
    print(f"\nFirst pair preview:")
    print(f"Resume length: {len(pairs[0][0])} characters")
    print(f"JD length: {len(pairs[0][1])} characters")
    
    # Test loading manual scores
    print("\nTesting manual scores loading:")
    manual_scores = load_manual_scores()
    if manual_scores:
        print(f"Manual scores range: {min(manual_scores)} to {max(manual_scores)}")
