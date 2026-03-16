"""
Module 1 — Automated Screening & Ranking
========================================
Ranks anonymized CVs against a Job Description (JD) using semantic similarity.

Technique:
    - Uses Sentence-Transformers (SBERT) to encode text into vectors.
    - Calculates Cosine Similarity between the JD vector and each CV vector.
    - Returns a ranked list of candidate IDs and their match scores (0.0 to 1.0).

Usage:
    from module1 import rank_candidates
    results = rank_candidates(cv_folder='anonymized_cvs', jd_text='...')
"""

import os
import logging
from typing import List, Tuple, Dict
import glob

# Try importing sentence_transformers; handle missing dependency gracefully
try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


logger = logging.getLogger(__name__)


class ScreenRanker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the SBERT model for semantic matching."""
        self.model = None
        if SBERT_AVAILABLE:
            logger.info("Loading SBERT model: %s ...", model_name)
            try:
                self.model = SentenceTransformer(model_name)
                logger.info("Model loaded successfully.")
            except Exception as e:
                logger.error("Failed to load SBERT model: %s", e)
        else:
            logger.warning("sentence-transformers not installed. Module 1 will fail.")

    def rank_candidates(
        self,
        anonymized_cv_dir: str,
        job_description: str
    ) -> List[Tuple[str, float]]:
        """
        Ranks all text files in the directory against the JD.

        Args:
            anonymized_cv_dir: Path to folder containing text CVs.
            job_description: The full text of the job description.

        Returns:
            List of tuples: [('Candidate_01', 0.85), ('Candidate_03', 0.72), ...]
            sorted by descending score.
        """
        if not self.model:
            logger.error("Ranking skipped: Model not loaded.")
            return []

        if not job_description.strip():
            logger.warning("Job description is empty.")
            return []

        # 1. Read all CV files
        cv_files = glob.glob(os.path.join(anonymized_cv_dir, "*.txt"))
        if not cv_files:
            logger.warning("No CVs found in %s", anonymized_cv_dir)
            return []

        cv_texts = []
        candidate_ids = []

        for filepath in cv_files:
            filename = os.path.basename(filepath)
            # Assuming filename format "Candidate_XX.txt"
            c_id = os.path.splitext(filename)[0]
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    cv_texts.append(text)
                    candidate_ids.append(c_id)
            except Exception as e:
                logger.warning("Could not read %s: %s", filename, e)

        if not cv_texts:
            return []

        # 2. Encode Job Description
        logger.info("Encoding Job Description...")
        jd_embedding = self.model.encode(job_description, convert_to_tensor=True)

        # 3. Encode CVs (batch processing)
        logger.info("Encoding %d CVs...", len(cv_texts))
        cv_embeddings = self.model.encode(cv_texts, convert_to_tensor=True)

        # 4. Compute Cosine Similarity
        # util.cos_sim returns a matrix [1, num_cvs]
        cosine_scores = util.cos_sim(jd_embedding, cv_embeddings)[0]

        # 5. Pair IDs with scores
        results = []
        for i, score in enumerate(cosine_scores):
            # Convert tensor to float
            results.append((candidate_ids[i], float(score)))

        # 6. Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info("Ranking complete. Top candidate: %s (Score: %.4f)", 
                    results[0][0], results[0][1])
        
        return results

# Singleton helper for easy import
_ranker = None

def load_ranker():
    global _ranker
    if _ranker is None:
        _ranker = ScreenRanker()
    return _ranker
