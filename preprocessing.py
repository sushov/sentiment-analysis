import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

# Load the dataset
data_path = 'data-set/train.csv'
df = pd.read_csv(data_path, encoding='latin-1')


# Access the text column in the dataset
text_data = df['SentimentText']

# Perform data preprocessing on each text entry
preprocessed_data = []
for text in text_data:
    # Text cleaning
    cleaned_text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)  # Remove URLs, mentions, and hashtags
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", cleaned_text)  # Remove special characters and numbers

    # Lowercasing
    cleaned_text = cleaned_text.lower()

    # Tokenization
    tokens = word_tokenize(cleaned_text)

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    preprocessed_text = ' '.join(stemmed_tokens)
    preprocessed_data.append(preprocessed_text)
    
    
# Create a new DataFrame with the preprocessed data
preprocessed_df = pd.DataFrame({'preprocessed_text': preprocessed_data})

# Save the preprocessed data to a new CSV file
output_file_path = 'data-set/preprocessed_data.csv'
preprocessed_df.to_csv(output_file_path, index=False)

# Perform sentiment analysis on the preprocessed data
# ...
# Rest of the sentiment analysis code
