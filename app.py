import streamlit as st
import joblib

# تحميل الموديل والـ vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("CV Screening System")
st.write("Upload or paste CV text to predict matching score")

# إدخال النص
text = st.text_area("Enter CV text here")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        st.success(f"Matched Score: {pred:.2f}")