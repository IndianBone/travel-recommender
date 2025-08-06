import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load TF-IDF vectorizer and data
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

df = pd.read_excel("processed_indian_places.xlsx")
print(df.columns.tolist())

tfidf_matrix = tfidf.transform(df['combined_features'])

# Streamlit UI
st.set_page_config(page_title="Travel Destination Recommender", layout="wide")

st.title("🏝️ Travel Destination Recommender")
st.write("Describe your ideal trip (e.g., *I want a historical place to visit in winter with good reviews and not too expensive*)")

user_input = st.text_area("Your travel preferences:", height=150)

if st.button("Recommend"):
    if user_input.strip() == "":
        st.warning("Please enter a travel description.")
    else:
        # Transform and calculate similarity
        user_vec = tfidf.transform([user_input])
        similarity_scores = cosine_similarity(user_vec, tfidf_matrix)
        top_indices = similarity_scores[0].argsort()[-5:][::-1]

        st.subheader("🌍 Top 5 Recommended Destinations")
        for idx in top_indices:
            name = df.iloc[idx]['Name']
            city = df.iloc[idx]['City'] if 'City' in df.columns else "Unknown"
            desc = df.iloc[idx]['combined_features']
            st.markdown(f"**{name}** - *{city}*  \n{desc}")
