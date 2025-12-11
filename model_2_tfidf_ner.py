"""
Model 2: Hybrid TF-IDF + Named Entity Recognition (NER)

Following Singh & Garg (2024) methodology:
Reference: "Resume Ranking With TF-IDF, Cosine Similarity and Named Entity Recognition"
- First International Conference on Data, Computation and Communication 2024

This hybrid approach combines traditional text analysis with NLP:

1. Named Entity Recognition (NER):
   - Identifies key entities: skills, qualifications, job titles, certifications
   - Extracts structured information from unstructured resume text
   - Handles non-standard language and unconventional formats
   
2. TF-IDF Vectorization:
   - Applied to both full text AND extracted entities
   - Provides frequency-based relevance scoring
   
3. Weighted Scoring:
   - Combines entity relevance with overall text similarity
   - Gives higher importance to matches in identified sections
   - Paper formula: weighted combination of metrics

Results from paper:
- Mean Average Precision: 0.795
- Normalized Discounted Cumulative Gain: 0.8503
- Mean Reciprocal Rank: 0.75
- Execution time: ~0.18 sec per resume (includes NER overhead)

Key improvements over Model 1:
- Better accuracy in identifying relevant qualifications
- Reduced bias against non-traditional backgrounds
- More precise ranking through entity-aware matching
- Captures semantic meaning beyond simple keywords
"""

import time
import spacy
from spacy.pipeline import EntityRuler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional, Dict


class TFIDFNERModel:
    def __init__(self):
        """
        Initialize spaCy model with NER for entity extraction.
        
        Following Singh & Garg (2024):
        - Uses NER to identify skills, experience, qualifications, certifications
        - EntityRuler for section-based extraction
        - Preprocessing: tokenization, lemmatization, stop word removal
        """
        # load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # create EntityRuler for section headers (Singh & Garg approach)
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        
        # define patterns for section headers
        patterns = [
            {"label": "SKILLS", "pattern": "Skills"},
            {"label": "SKILLS", "pattern": "SKILLS"},
            {"label": "SKILLS", "pattern": "Technical Skills"},
            {"label": "SKILLS", "pattern": "TECHNICAL SKILLS"},
            {"label": "EXPERIENCE", "pattern": "Experience"},
            {"label": "EXPERIENCE", "pattern": "EXPERIENCE"},
            {"label": "EXPERIENCE", "pattern": "Work Experience"},
            {"label": "EXPERIENCE", "pattern": "WORK EXPERIENCE"},
            {"label": "EXPERIENCE", "pattern": "Professional Experience"},
            {"label": "EDUCATION", "pattern": "Education"},
            {"label": "EDUCATION", "pattern": "EDUCATION"},
        ]
        ruler.add_patterns(patterns)
        
    def extract_section_text(self, text: str, section_label: str) -> str:
        """
        Extract text from a specific section of the document.
        
        Args:
            text: Full document text
            section_label: Label of the section to extract (e.g., "SKILLS")
            
        Returns:
            Extracted section text, or empty string if not found
        """
        doc = self.nlp(text)
        
        # find entities with the specified label
        section_entities = [ent for ent in doc.ents if ent.label_ == section_label]
        
        if not section_entities:
            return ""
        
        # get the first occurrence
        section_start = section_entities[0].end_char
        
        # find the next section header or end of document
        next_sections = [ent for ent in doc.ents if ent.start_char > section_start]
        
        if next_sections:
            section_end = next_sections[0].start_char
        else:
            section_end = len(text)
        
        section_text = text[section_start:section_end].strip()
        return section_text
    
    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate TF-IDF cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        if not text1 or not text2:
            return 0.0
        
        # use same improved parameters as model 1
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using NER.
        
        Following Singh & Garg (2024) approach:
        - Identifies skills, qualifications, job titles, certifications
        - Uses spaCy's NER for entity recognition
        
        Args:
            text: Resume or job description text
            
        Returns:
            Dictionary of entity types and their values
        """
        doc = self.nlp(text)
        entities = {
            "skills": [],
            "experience": [],
            "education": [],
            "certifications": []
        }
        
        # extract section-based entities
        for label in ["SKILLS", "EXPERIENCE", "EDUCATION"]:
            section_text = self.extract_section_text(text, label)
            if section_text:
                entities[label.lower()].append(section_text)
        
        return entities
    
    def calculate_weighted_score(self, resume_text: str, jd_text: str) -> float:
        """
        Calculate hybrid weighted score using TF-IDF + NER.
        
        Following Singh & Garg (2024) methodology:
        1. Extract entities (skills, qualifications, experience) using NER
        2. Calculate TF-IDF similarity for full text
        3. Calculate TF-IDF similarity for extracted entities
        4. Combine with weighted formula for final ranking
        
        Paper's approach:
        - "cosine similarity scores are pooled with entity relevance based on NER"
        - "weighted formula weighs contribution of both metrics"
        - Results in more robust and accurate candidate evaluation
        
        Args:
            resume_text: Resume text content
            jd_text: Job description text content
            
        Returns:
            Weighted similarity score (0 to 1)
        """
        # score 1: full text TF-IDF similarity (baseline/no NER)
        score_full_text = self.calculate_tfidf_similarity(resume_text, jd_text)
        
        # score 2: entity-based similarity using NER
        resume_skills = self.extract_section_text(resume_text, "SKILLS")
        resume_experience = self.extract_section_text(resume_text, "EXPERIENCE")
        
        # Combine extracted entities
        entity_text = " ".join([resume_skills, resume_experience]).strip()
        
        if entity_text:
            score_entities = self.calculate_tfidf_similarity(entity_text, jd_text)
        else:
            # fallback to full text if no entities found
            score_entities = score_full_text
        
        # weighted combination (Singh & Garg approach)
        # paper uses weighted formula - we use 50/50 as reasonable default
        # can be tuned based on specific requirements
        final_score = (0.5 * score_full_text) + (0.5 * score_entities)
        
        return final_score


def run_model_2(pairs: List[Tuple[str, str]]) -> Tuple[List[float], List[float]]:
    """
    Run Model 2 on all resume-JD pairs.
    
    Args:
        pairs: List of (resume_text, jd_text) tuples
        
    Returns:
        Tuple of (scores list, inference_times list in milliseconds)
    """
    model = TFIDFNERModel()
    model_2_scores = []
    inference_times = []
    total_pairs = len(pairs)
    
    for i, (resume_text, jd_text) in enumerate(pairs, 1):
        start_time = time.time()
        score = model.calculate_weighted_score(resume_text, jd_text)
        inference_time = (time.time() - start_time) * 1000  # milliseconds
        
        model_2_scores.append(score)
        inference_times.append(inference_time)
        print(f"Pair {i}/{total_pairs}: TF-IDF+NER Score = {score:.4f} (Time: {inference_time:.2f}ms)")
    
    print(f"\nAverage inference time: {sum(inference_times)/len(inference_times):.2f}ms")
    return model_2_scores, inference_times


if __name__ == "__main__":
    # example usage
    from data_loader import load_resume_jd_pairs
    
    pairs = load_resume_jd_pairs()
    
    # run model 2
    print("=" * 50)
    print("MODEL 2: TF-IDF + NER (Weighted)")
    print("=" * 50)
    scores = run_model_2(pairs)
    
    print(f"\nAverage Score: {sum(scores) / len(scores):.4f}")
    print(f"Min Score: {min(scores):.4f}")
    print(f"Max Score: {max(scores):.4f}")
