# Review Insights: Ask Questions About Customer Reviews

A small web app where you upload a CSV of customer reviews, type a question in plain English, and get an AI-generated answer grounded in the most relevant reviews. Built with Streamlit and the Google Gemini API.

## Business Question
Can someone get a specific, evidence-backed answer about customer feedback without reading every review, and without trusting an AI to guess from general knowledge?

## How It Works
This app uses Retrieval-Augmented Generation (RAG), built from scratch:
1. **Embed**: each review is converted into an embedding (a list of numbers capturing its meaning) using Gemini's embedding model
2. **Retrieve**: the question is embedded the same way, and cosine similarity finds the reviews closest in meaning
3. **Generate**: only the top matching reviews are passed to an LLM, which is told to answer based only on that text
4. **Show the evidence**: the app displays the exact reviews used, along with their similarity scores, so the answer can be checked

## Tools Used
Python, Streamlit, pandas, numpy, Google Gemini API (`gemini-embedding-001` and `gemini-3.5-flash-lite`)

## Run It Locally
1. Install the requirements: `pip install -r requirements.txt`
2. Start the app: `streamlit run app.py`
3. Get a free API key from Google AI Studio (aistudio.google.com) and paste it into the app
4. Upload a reviews CSV that has a `Review Text` column and ask a question

The API key is typed into the app at runtime and is never saved in the code or in any file.

## Example
The dataset used for testing is the Women's E-Commerce Clothing Reviews dataset on Kaggle (CC0 Public Domain).

**Question:** What do customers complain about with sizing?

**Result:** The app retrieved reviews about restricted arm movement and items running small, then answered using those specific details.

## Limitations
- To stay within free-tier API quotas, the app embeds a random sample of 15 reviews, so answers reflect that sample rather than the full dataset
- Embeddings are held in memory rather than in a dedicated vector database. The retrieval logic is the same as a production system, but a vector database would be needed to search large datasets quickly
- Retrieval can pull in loosely related reviews (for example, a shrinking-in-the-wash review for a sizing question), so the app shows the source reviews for transparency

## Files
- `app.py`: the Streamlit app
- `requirements.txt`: Python packages needed to run it
- `Womens Clothing E-Commerce Reviews.csv`: sample dataset for trying the app (CC0 Public Domain)
