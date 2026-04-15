import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# with open("vectorizer.pkl", "rb") as file:
#     vectorizer = pickle.load(file)

# with open("model.pkl", "rb") as file:
#     LR = pickle.load(file)

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
LR = pickle.load(open('model.pkl', 'rb'))

# pickle.load(LR, open("model.pkl", "wb"))
# pickle.load(vectorization, open("vectorizer.pkl", "wb"))

# Text cleaning function
def wordopt(text):
    return text.lower()

# Label mapping
def output_label(n):
    return "Fake News" if n == 0 else "Real News"

# UI
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title(" Fake News Detector")
st.write("Enter news text below to check whether it is Fake or Real")

# Input box
user_input = st.text_area("Enter News Text")

# Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned_text = wordopt(user_input)
        vectorized = vectorizer.transform([cleaned_text])
        prediction = LR.predict(vectorized)

        # st.success(f"Prediction: {output_label(prediction)}")
        if prediction == 0:
                st.error("🚨 Fake News")
        else:
                st.success("✅ Real News")

