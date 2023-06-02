import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def load_model(model_path):
    # Load the saved model
    model = joblib.load(model_path)
    return model

def train_model():
    # Load the dataset
    data_path = 'data-set/train.csv'
    df = pd.read_csv(data_path, encoding='latin1')

    # Print the column names
    print(df.columns)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['SentimentText'], df['Sentiment'], test_size=0.2, random_state=42)

    # Convert X_train to string type
    X_train = X_train.astype(str)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the training data
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Save the vectorizer model
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_path)
    print('Vectorizer model saved.')

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)

    # Evaluate the model on the testing data
    X_test_vectorized = vectorizer.transform(X_test)
    accuracy = model.score(X_test_vectorized, y_test)
    print('Accuracy:', accuracy)

    # Save the model
    model_path = 'models/sentiment_analysis_model.pkl'
    joblib.dump(model, model_path)
    print('Model saved.')

if __name__ == '__main__':
    train_model()
