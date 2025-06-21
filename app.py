import streamlit as st
import sys
import subprocess
import importlib

# Function to install missing packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install required packages
required_packages = {
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'nltk': 'nltk'
}

for import_name, package_name in required_packages.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        with st.spinner(f"Installing required package: {package_name}..."):
            install_package(package_name)
        importlib.import_module(import_name)

# Now safely import everything
import numpy as np
import random
from collections import Counter
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import nltk

# Download NLTK data
try:
    nltk.data.find('corpora/brown')
except LookupError:
    with st.spinner("Downloading NLTK data..."):
        nltk.download('brown')


def tokenize(text):
    return re.findall(r'\w+', text.lower())

def build_vocab(corpus, vocab_size=1000):
    words = tokenize(corpus)
    word_counts = Counter(words)
    common_words = word_counts.most_common(vocab_size)
    vocab = {word: idx for idx, (word, _) in enumerate(common_words)}
    return vocab

def encode(text, vocab):
    words = tokenize(text)
    return [vocab.get(word, -1) for word in words if word in vocab]

class MultimodalKeyboardDataset:
    def __init__(self, text_data, typing_speeds):
        self.text_data = text_data
        self.typing_speeds = typing_speeds
    
    def __len__(self):
        return len(self.text_data) - 1
    
    def __getitem__(self, idx):
        return {
            'text': self.text_data[idx],
            'speed': self.typing_speeds[idx],
            'target': self.text_data[idx + 1]
        }

def prepare_data(corpus, vocab):
    words = tokenize(corpus)
    encoded_words = [vocab.get(word, -1) for word in words if word in vocab]
    
    # Generate random typing speeds for demo purposes
    typing_speeds = [random.uniform(0.1, 1.0) for _ in range(len(encoded_words))]
    
    dataset = MultimodalKeyboardDataset(encoded_words, typing_speeds)
    
    X = np.array([[x['text'], x['speed']] for x in dataset])
    y = np.array([x['target'] for x in dataset])
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(corpus, model_type='random_forest'):
    vocab = build_vocab(corpus)
    X_train, X_test, y_train, y_test = prepare_data(corpus, vocab)
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == 'neural_net':
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100)
    else:
        raise ValueError("Invalid model type")
    
    model.fit(X_train, y_train)
    return model, vocab

def predict_next_word(model, vocab, input_text, typing_speeds):
    words = tokenize(input_text)
    encoded_words = [vocab.get(word, -1) for word in words if word in vocab]
    
    if not encoded_words:
        return random.choice(list(vocab.keys()))
    
    # Use the last word and average typing speed
    last_word = encoded_words[-1]
    avg_speed = sum(typing_speeds) / len(typing_speeds) if typing_speeds else 0.5
    
    prediction = model.predict([[last_word, avg_speed]])
    predicted_idx = prediction[0]
    
    # Map index back to word
    word_lookup = {idx: word for word, idx in vocab.items()}
    return word_lookup.get(predicted_idx, "unknown")

def main():
    st.title("🧠 Predictive Keyboard with Typing Speed")
    corpus = ' '.join(brown.words()[:10000])
    
    model_type = st.sidebar.selectbox("Select Model", ['random_forest', 'neural_net'])
    
    with st.spinner('Training model...'):
        model, vocab = train_model(corpus, model_type)
    
    col1, col2 = st.columns(2)
    with col1:
        input_text = st.text_input("Enter text:", "predictive keyboard model using")
    with col2:
        typing_input = st.text_input("Typing speeds (comma-separated):", "0.5, 0.6, 0.4, 0.3")

    if st.button("Predict Next Word"):
        try:
            speeds = list(map(float, typing_input.split(',')))
            predicted_word = predict_next_word(model, vocab, input_text, speeds)
            st.success(f"Predicted: **{predicted_word}**")
        except ValueError:
            st.error("Please enter valid numbers for typing speeds")

if __name__ == "__main__":
    main()
