from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two preprocessed texts
    using TF-IDF vectorization.

    Returns a float between 0.0 and 1.0 (0% – 100% match).
    """
    if not text1.strip() or not text2.strip():
        return 0.0

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(score[0][0])


def match_percentage(text1: str, text2: str) -> float:
    """Return similarity as a percentage (0–100), rounded to 1 decimal."""
    return round(compute_similarity(text1, text2) * 100, 1)
