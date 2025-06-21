import streamlit as st
import numpy as np
import random
from collections import Counter
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import brown

# Download NLTK data
try:
    nltk.data.find('corpora/brown')
except LookupError:
    with st.spinner("Downloading NLTK data..."):
        nltk.download('brown')

# [PASTE ALL YOUR EXISTING FUNCTIONS HERE]
# Include: tokenize(), build_vocab(), encode(), MultimodalKeyboardDataset,
# prepare_data(), train_model(), predict_next_word()

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
