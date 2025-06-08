import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/logistic_model.joblib")
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")

# App UI
st.title("🩺 Health App Review Sentiment Analyzer")

user_input = st.text_area("Enter a review of a health app:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess and vectorize
        input_vec = vectorizer.transform([user_input])
        
        # Predict
        pred = model.predict(input_vec)[0]
        
        # Map label to sentiment
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.markdown(f"**Predicted Sentiment:** `{label_map[pred]}`")
