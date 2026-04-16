import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📊",
    layout="wide"
)

# Load model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Sentiment", "Visualization"])

# Home Page
if page == "Home":
    st.title("📊 Twitter Sentiment Analysis")
    st.write("""
    Welcome to the Sentiment Analysis Web App.
    
    This application predicts whether text sentiment is:
    - 😊 Positive
    - 😐 Neutral
    - 😞 Negative
    """)

# Prediction Page
elif page == "Predict Sentiment":
    st.title("🧠 Sentiment Prediction")

    user_input = st.text_area("Enter your tweet or text here")

    if st.button("Predict"):
        if user_input.strip() != "":
            transformed_text = vectorizer.transform([user_input])
            prediction = model.predict(transformed_text)[0]

            if prediction == 1:
                st.success("😊 Positive Sentiment")
            elif prediction == 0:
                st.info("😐 Neutral Sentiment")
            else:
                st.error("😞 Negative Sentiment")
        else:
            st.warning("Please enter some text.")

# Visualization Page
elif page == "Visualization":
    st.title("📈 Sentiment Dashboard")

    df = pd.read_csv("Twitter_Data.csv")

    sentiment_counts = df['category'].value_counts()

    fig, ax = plt.subplots()
    ax.bar(['Negative', 'Neutral', 'Positive'], sentiment_counts)
    st.pyplot(fig)

    text = " ".join(df['clean_text'].dropna().astype(str).tolist()[:1000])

    wordcloud = WordCloud(width=800, height=400).generate(text)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wordcloud)
    ax2.axis("off")

    st.pyplot(fig2)