from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import re
nlp = spacy.load("en_core_web_sm")

def clean_string(text, stem="lem"):
    # lower case
    text = text.lower()
    # tokenization
    tokens = nlp(text)
    cleaned_str = []
    for token in tokens:
        # trimming
        text = token.text.strip()
        # remove unicode characters
        text = re.sub(r'\W', '', text)
        # Stemming
        if stem == "stem":
            stemmer = PorterStemmer()
            token = stemmer.stem(text)
        # Lemmatization
        elif stem == 'lem':
            limatizer = WordNetLemmatizer()
            token = limatizer.lemmatize(text)
        cleaned_str.append(text)
    return " ".join(cleaned_str)