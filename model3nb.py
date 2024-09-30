import pandas as pd
import re
from nltk import pos_tag
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score  # Import accuracy_score

import nltk
nltk.download('averaged_perceptron_tagger')

# Install necessary packages
# !pip install nltk scikit-learn pandas

# Load the data (replace with the actual loading mechanism)
def load_data():
    pos_reviews = open('rt-polaritydata/rt-polarity.pos', 'r', encoding='ISO-8859-1').readlines()
    neg_reviews = open('rt-polaritydata/rt-polarity.neg', 'r', encoding='ISO-8859-1').readlines()
    return pos_reviews, neg_reviews

# Split data
def split_data(pos_reviews, neg_reviews):
    train_pos, valid_pos, test_pos = pos_reviews[:4000], pos_reviews[4000:4500], pos_reviews[4500:]
    train_neg, valid_neg, test_neg = neg_reviews[:4000], neg_reviews[4000:4500], neg_reviews[4500:]
    
    train_data = train_pos + train_neg
    valid_data = valid_pos + valid_neg
    test_data = test_pos + test_neg
    train_labels = [1]*4000 + [0]*4000
    valid_labels = [1]*500 + [0]*500
    test_labels = [1]*831 + [0]*831
    
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

# POS Tagging (Part-of-Speech)
class POSTagger(TransformerMixin):
    def transform(self, X, *_):
        return [' '.join([f'{word}_{pos}' for word, pos in pos_tag(sentence.split())]) for sentence in X]
    
    def fit(self, *_):
        return self

# Preprocessing (lowercasing, punctuation removal, negation handling)
def preprocess_text(text):
    text = text.lower()
    # text = handle_negation(text)  # Handle negations
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply preprocessing to all data
def preprocess_data(data):
    return [preprocess_text(text) for text in data]

# Create the Naive Bayes model pipeline with optimizations
def create_model():
    model_pipeline = Pipeline([
        ('pos', POSTagger()),  # POS Tagging
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), binary=True)),  # Binary TF-IDF with unigram+bigram
        ('nb', MultinomialNB())  # Naive Bayes Classifier
    ])
    return model_pipeline

# Hyperparameter Tuning (optional for Naive Bayes, but we can search for alpha parameter)
def tune_hyperparameters(train_data, train_labels):
    model_pipeline = create_model()
    param_grid = {
        'nb__alpha': [0.5, 1.0, 2.0]  # Adjust alpha for Naive Bayes smoothing
    }
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3)
    grid_search.fit(train_data, train_labels)
    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_



# Training and Evaluation
def train_and_evaluate(train_data, train_labels, test_data, test_labels):
    # Preprocess the data
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    
    best_model = tune_hyperparameters(train_data, train_labels)

    best_model.fit(train_data, train_labels)

    predictions = best_model.predict(test_data)
    
    accuracy = accuracy_score(test_labels, predictions)  # Calculate accuracy
    
    cm = confusion_matrix(test_labels, predictions)
    report = classification_report(test_labels, predictions, target_names=['Negative', 'Positive'])
    
    return cm, report, accuracy  

# Main execution
if __name__ == "__main__":
    pos_reviews, neg_reviews = load_data()
    
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = split_data(pos_reviews, neg_reviews)
    
    cm, report, accuracy = train_and_evaluate(train_data, train_labels, test_data, test_labels)
    
    # Print results
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    print("Accuracy: {:.2f}%".format(accuracy * 100))  # Print accuracy as a percentage
