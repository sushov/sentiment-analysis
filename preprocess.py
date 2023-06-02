import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    cleaned_text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", cleaned_text)
    cleaned_text = cleaned_text.lower()
    tokens = word_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text
