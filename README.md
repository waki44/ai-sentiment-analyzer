# ai-sentiment-analyzer
An end-to-end Machine Learning web application that classifies customer reviews into positive or negative sentiments using Scikit-Learn, TF-IDF, and Streamlit.
# 📊 AI Review Sentiment Analysis & Classification Engine

A production-ready Natural Language Processing (NLP) web system engineered to perform real-time sentiment classification on unstructured customer reviews. The application leverages a Scikit-Learn machine learning pipeline, TF-IDF vectorization, and a responsive Streamlit interface for live text inference.

---

## 📌 Executive Summary

Understanding customer sentiment at scale is critical for data-driven product decisions. This project delivers an end-to-end NLP solution that transforms raw text input into vectorized feature spaces, executes probabilistic classification via Naive Bayes, and serves real-time predictions through an interactive web UI.

---

## 🚀 Key Highlights & Engineering Features

- **Text Vectorization Pipeline:** Employs `TfidfVectorizer` with n-gram capabilities to penalize ubiquitous terms and extract high-signal semantic tokens.
- **Multinomial Naive Bayes Model:** Utilizes a probabilistic classification engine optimized for high-dimensional sparse text data.
- **Atomic ML Pipeline:** Bundles vectorization and classification into an integrated Scikit-Learn `Pipeline` to eliminate data leakage and ensure uniform inference transformations.
- **Model Serialization & Artifact Storage:** Serializes the trained pipeline into `.pkl` format using Python's `pickle` library for low-latency loading.
- **Interactive UI with State Caching:** Deploys a clean `Streamlit` dashboard using `@st.cache_resource` to minimize memory overhead during model inference.

---

## 🛠️ Architecture & Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.12+ |
| **Machine Learning / NLP** | Scikit-Learn, NumPy, Pandas |
| **Model Persistence** | Pickle |
| **Frontend Framework** | Streamlit |
| **Version Control** | Git & GitHub |

---

## 📂 Repository Structure

```text
├── ai sentiment.py       # Data preparation, pipeline fitting, and artifact export
├── app.py                # Streamlit UI with cached model loading and prediction handlers
├── sentiment_model.pkl   # Serialized ML pipeline artifact (TF-IDF + MultinomialNB)
├── requirements.txt      # Pinned project dependencies
└── README.md             # Technical documentation
