from pickles import *
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lower case
    text = text.lower()
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove punctuation and stop words, and lemmatize the tokens
    cleaned_tokens = [
        lemmatizer.lemmatize(token) 
        for token in tokens 
        if token not in string.punctuation and token not in stop_words
    ]
    
    return ' '.join(cleaned_tokens)

def vectorize(clean_data):
    # Transform the cleaned text to TF-IDF representation using the loaded vectorizer
    text_tfidf = vectorizer.transform([clean_data])

    return text_tfidf

def predict(tf_data):

    # Make prediction using the loaded model
    prediction = model.predict(tf_data)

    # The labels to the data
    labels = label_categories

    return labels[prediction[0]]

def execute_flow(text):
    clean_texts = preprocess_text(text)
    vectorize_data = vectorize(clean_data=clean_texts)
    prediction = predict(vectorize_data)
    return prediction