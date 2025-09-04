# Advanced Sentiment Analysis on Twitter Posts using NLP and ML

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (this will create `sentiment_model.pkl` and `vectorizer.pkl`):
   ```bash
   python train_model.py
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 📂 Files Included
- `app.py` → Streamlit web app for single tweet & bulk analysis with wordclouds.
- `train_model.py` → Trains a Naive Bayes model and saves it.
- `sample_tweets.csv` → Example dataset for training/testing.
- `requirements.txt` → Dependencies list.
- `README.md` → Instructions.

## ✅ Features
- Single tweet prediction.
- Bulk CSV upload with sentiment results.
- Wordclouds for positive, negative, and neutral tweets.
- Downloadable CSV results.
