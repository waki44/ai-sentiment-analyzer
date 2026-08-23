import streamlit as st
import pickle

st.set_page_config(page_title="AI Review Analyzer", layout="centered")

st.title("📊 AI Review Sentiment Analyzer")
st.write("Enter customer review text below to classify sentiment.")

# Load Model
@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# User Input
user_review = st.text_area("Review Text", placeholder="Write something like: Excellent product...")

if st.button("Analyze Sentiment"):
    if user_review.strip():
        prediction = model.predict([user_review])[0]
        if prediction == "positive":
            st.success("Result: **Positive Review** 👍")
        else:
            st.error("Result: **Negative Review** 👎")
    else:
        st.warning("Please enter some text first.")