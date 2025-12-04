import re
import spacy
from collections import Counter
import nltk

# Initialize NLP model and other components
try:
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    raise RuntimeError(f"Failed to load spaCy model: {e}")

try:
    nltk.download('stopwords')
    nltk.download('punkt')
except Exception as e:
    raise RuntimeError(f"Failed to download NLTK resources: {e}")


def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by cleaning and lemmatizing it.
    """
    try:
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)

        doc = nlp(text)
        return ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)
    except Exception as e:
        raise RuntimeError(f"Error during text preprocessing: {e}")


def identify_company_name(texts):
    """
    Identifies the most common company name from a list of texts.
    """
    try:
        all_entities = []
        for text in texts:
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
            all_entities.extend(entities)

        if all_entities:
            most_common = Counter(all_entities).most_common(1)
            return most_common[0][0]

        return 'Unknown'
    except Exception as e:
        raise RuntimeError(f"Error during company name identification: {e}")