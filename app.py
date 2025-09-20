import streamlit as st
import joblib
import os

# Load trained pipeline (trainable_file.joblib)
model_path = os.path.join(os.path.dirname(file), "trainable_file.joblib")
pipeline = joblib.load(model_path)

st.set_page_config(page_title="IMDB Sentiment Tester", layout="centered")
st.title("IMDB Sentiment Tester")
st.write("Enter a movie review below to predict its sentiment.")

with st.form("sentiment_form"):
    review_text = st.text_area(
        "Movie Review Text:",
        "The movie was fantastic, with a gripping storyline and excellent acting.",
        height=150
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    if not review_text.strip():
        st.error("Please enter a review text.")
    else:
        try:
            pred_label = pipeline.predict([review_text])[0]
            sentiment = "Positive" if pred_label == 1 else "Negative"
            st.markdown(f"*Prediction:* :{'green' if sentiment=='Positive' else 'red'}[{sentiment}]")
        except Exception as e:
            st.error(f"Error: {e}")

