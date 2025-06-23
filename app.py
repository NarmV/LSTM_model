import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import re

# Load model and tokenizer only once using cache
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_new.keras")

@st.cache_resource
def load_tokenizer():
    return joblib.load('Tokeniser.pkl')

model = load_lstm_model()
tokenizer = load_tokenizer()
max_length = 100  # Set during training

# Text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text.lower()

# Prediction function with error handling
def predict_sentiment(text):
    try:
        text = clean_text(text)
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_length, padding='post')
        prediction = model.predict(padded)[0]
        sentiment = np.argmax(prediction)
        confidence = float(prediction[sentiment])
        label = "Positive 😊" if sentiment == 1 else "Negative 😞"
        return label, confidence, prediction
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Error", 0.0, [0.0, 0.0]

# UI layout
st.title("LSTM Sentiment Classifier App")
st.write("Enter your review text below:")

# Input and buttons
col1, col2 = st.columns([2, 1])
with col1:
    user_input = st.text_input("", key="input_text")
with col2:
    clear = st.button("Clear")
    if clear:
        st.session_state.input_text = ""

# Predict button logic
if st.button("Predict") and user_input:
    label, confidence, probabilities = predict_sentiment(user_input)
    if label != "Error":
        st.write(f"**Predicted Sentiment:** {label}")
        st.write(f"**Confidence Score:** {confidence:.2f}")
        st.write("### All Class Probabilities:")
        st.write(f"Negative 😞: {probabilities[0]:.2f}")
        st.write(f"Positive 😊: {probabilities[1]:.2f}")
