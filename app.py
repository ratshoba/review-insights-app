import streamlit as st
import pandas as pd
import numpy as np
import time
from google import genai
from google.genai import types

st.title("Review Insights")
st.write("Upload a CSV of customer reviews and ask a question about them.")

api_key = st.text_input("Enter your Gemini API key", type="password")
uploaded_file = st.file_uploader("Upload your reviews CSV", type="csv")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if uploaded_file is not None and api_key:
    df = pd.read_csv(uploaded_file, encoding="latin1")
    df = df.dropna(subset=["Review Text"])
    st.write("Preview of your data:")
    st.dataframe(df.head())

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30000))

    # Reset saved embeddings if a different file is uploaded
    if st.session_state.get("file_name") != uploaded_file.name:
        st.session_state.pop("sample", None)
        st.session_state.pop("embeddings", None)
        st.session_state["file_name"] = uploaded_file.name

    try:
        # Embed a sample of reviews once, then reuse it for every question
        if "sample" not in st.session_state:
            with st.spinner("Preparing reviews (one-time step, about 30 seconds)..."):
                sample = df.sample(min(15, len(df)), random_state=1).copy()
                embeddings = []
                for text in sample["Review Text"]:
                    result = client.models.embed_content(
                        model="gemini-embedding-001", contents=text
                    )
                    embeddings.append(result.embeddings[0].values)
                    time.sleep(2)
                st.session_state["sample"] = sample
                st.session_state["embeddings"] = embeddings

        question = st.text_input("Ask a question about these reviews")

        if question:
            with st.spinner("Searching reviews and generating an answer..."):
                q_result = client.models.embed_content(
                    model="gemini-embedding-001", contents=question
                )
                q_embedding = q_result.embeddings[0].values

                sample = st.session_state["sample"].copy()
                sample["similarity"] = [
                    cosine_similarity(q_embedding, e)
                    for e in st.session_state["embeddings"]
                ]
                top = sample.sort_values("similarity", ascending=False).head(4)

                context = "\n\n".join([f"Review: {r}" for r in top["Review Text"]])
                prompt = (
                    "Based only on the following customer reviews, "
                    f"answer this question: {question}\n\n{context}"
                )
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite", contents=prompt
                )

            st.subheader("Answer")
            st.write(response.text)
            st.subheader("Reviews used to generate this answer")
            st.dataframe(top[["Review Text", "similarity"]])

    except Exception as e:
        st.error(f"Something went wrong: {e}")

elif uploaded_file is not None and not api_key:
    st.info("Enter your API key above to continue.")