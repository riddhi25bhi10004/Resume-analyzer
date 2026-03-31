import re
import nltk
import spacy

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")

STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Lowercase, remove special characters, extra spaces."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_text(text: str) -> str:
    """Lemmatize and remove stopwords using spaCy."""
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in STOP_WORDS and not token.is_punct and not token.is_space
    ]
    return ' '.join(tokens)


def preprocess(text: str) -> str:
    """Full pipeline: clean → lemmatize."""
    return lemmatize_text(clean_text(text))
