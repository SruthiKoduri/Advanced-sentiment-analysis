import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load trained model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Advanced Sentiment Analysis on Twitter Posts using NLP and ML")

menu = ["Single Tweet", "Bulk Upload"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Single Tweet":
    tweet = st.text_area("Enter a tweet for sentiment analysis")
    if st.button("Predict"):
        if tweet.strip() != "":
            vect_tweet = vectorizer.transform([tweet])
            prediction = model.predict(vect_tweet)[0]
            st.success(f"Predicted Sentiment: {prediction}")
        else:
            st.warning("Please enter a tweet")

elif choice == "Bulk Upload":
    uploaded_file = st.file_uploader("Upload a CSV file with a 'tweet' column", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "tweet" in df.columns:
            vect_tweets = vectorizer.transform(df["tweet"].astype(str))
            predictions = model.predict(vect_tweets)
            df["Predicted_Sentiment"] = predictions
            st.dataframe(df.head())

            # Wordclouds for each sentiment
            for sentiment in ["positive", "negative", "neutral"]:
                text = " ".join(df[df["Predicted_Sentiment"] == sentiment]["tweet"].astype(str))
                if text.strip():
                    wc = WordCloud(width=600, height=400, background_color="white").generate(text)
                    st.subheader(f"WordCloud for {sentiment} tweets")
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)

            # Download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "results.csv", "text/csv")
        else:
            st.error("CSV must contain a 'tweet' column")
