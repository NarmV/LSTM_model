import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import re

# Load model and tokenizer
model = load_model("lstm_new.keras")
tokenizer = joblib.load('tokenizer.pkl')

max_length = 100  # As per your training

def clean_text(text):
    # Basic cleaning same as training preprocessing
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text.lower()

def predict_sentiment(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    prediction = model.predict(padded)[0]
    
    sentiment = np.argmax(prediction)
    confidence = prediction[sentiment]

    label = "Positive 😊" if sentiment == 1 else "Negative 😞"

    return label, confidence, prediction

st.title("LSTM Sentiment Classifier App")

user_input = st.text_input("Enter your review text:")

if st.button("Predict"):
    if user_input:
        label, confidence, probabilities = predict_sentiment(user_input)
        st.write(f"**Predicted Sentiment:** {label}")
        st.write(f"**Confidence Score:** {confidence:.2f}")

        st.write("### All Class Probabilities:")
        st.write(f"Negative 😞: {probabilities[0]:.2f}")
        st.write(f"Positive 😊: {probabilities[1]:.2f}")
