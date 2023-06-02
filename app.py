from flask import Flask, render_template, request
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import sys

nltk.download('wordnet')
nltk.download('omw-1.4')


app = Flask(__name__)

def preprocess_text(text):
    # Remove punctuation
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a single string
    processed_text = ' '.join(tokens)

    return processed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Load the saved model
    model_path = 'models/sentiment_analysis_model.pkl'
    model = joblib.load(model_path)

    # Get the input text from the user
    text = request.form['text']

    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Vectorize the preprocessed text
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    vectorizer = joblib.load(vectorizer_path)
    preprocessed_text_vectorized = vectorizer.transform([preprocessed_text])

    # Make the prediction
    sentiment = model.predict(preprocessed_text_vectorized)[0]
    print(sentiment)
    SystemExit

    return render_template('result.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
