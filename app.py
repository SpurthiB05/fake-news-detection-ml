import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake News Detection AI")
st.write("Enter a news headline to check if it is Fake or Real")

news = st.text_input("Enter news headline")

if st.button("Predict"):
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0] == "fake":
        st.error("This news is FAKE")
    else:
        st.success("This news is REAL")