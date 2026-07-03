"""
Streamlit Dashboard -- The E-Commerce Shield
De-Influencing & Anti-Consumerism Discourse Intelligence Platform

Run with:  streamlit run dashboard_app.py

Required files in the same working directory (outputs from Notebooks 2-6):
    ner_comments.csv.gz          (Notebook 5)
    ner_entities.csv.gz          (Notebook 5)
    keywords_overall.csv         (Notebook 6)
    keywords_by_perspective.csv  (Notebook 6)
    keywords_by_sentiment.csv    (Notebook 6)
    tfidf_vectorizer.pkl         (Notebook 6)
    tfidf_matrix.pkl             (Notebook 6)
    tfidf_row_index.csv          (Notebook 6)

Groq API key: set as an environment variable GROQ_API_KEY, or paste it into
the sidebar field when the app is running.
"""

import os
import ast
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ────────────────────────────────────────────────────────────────────────
# Page config & styling
# ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The E-Commerce Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0f1117; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1a1c29 0%, #2b2140 100%);
        margin-bottom: 1.5rem;
        border: 1px solid #33364a;
    }
    .hero h1 { color: #ffffff; margin-bottom: 0.25rem; font-size: 2.1rem; }
    .hero p { color: #b8bcd4; font-size: 1rem; margin: 0; }
    div[data-testid="stMetric"] {
        background: #1a1c29;
        border: 1px solid #33364a;
        border-radius: 12px;
        padding: 1rem;
    }
    div[data-testid="stMetricLabel"] { color: #b8bcd4; }
    .chat-answer {
        background: #1a1c29;
        border-left: 4px solid #7c5cff;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }
    .source-chip {
        display: inline-block;
        background: #2b2140;
        color: #d8b4fe;
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        font-size: 0.75rem;
        margin-right: 0.4rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PERSPECTIVE_COLORS = px.colors.qualitative.Bold
SENTIMENT_COLORS = {"Negative": "#e05263", "Neutral": "#8a8f9e", "Positive": "#4caf7d"}


# ────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_core_data():
    df = pd.read_csv("ner_comments.csv.gz")
    df = df.dropna(subset=["light_clean_text"]).reset_index(drop=True)
    df["light_clean_text"] = df["light_clean_text"].astype(str)
    df["entities"] = df["entities"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    return df


@st.cache_data(show_spinner="Loading entity table...")
def load_entities():
    return pd.read_csv("ner_entities.csv.gz")


@st.cache_data(show_spinner="Loading keyword tables...")
def load_keywords():
    overall = pd.read_csv("keywords_overall.csv")
    by_perspective = pd.read_csv("keywords_by_perspective.csv")
    by_sentiment = pd.read_csv("keywords_by_sentiment.csv")
    return overall, by_perspective, by_sentiment


# ────────────────────────────────────────────────────────────────────────
# RAG system (cached resource -- built once per session)
# ────────────────────────────────────────────────────────────────────────
class FastSemanticEmbeddings:
    """TF-IDF + TruncatedSVD (LSA) embeddings -- fast, no neural inference needed."""
    def __init__(self, texts, n_components=100):
        self.vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, stop_words="english")
        tfidf = self.vectorizer.fit_transform(texts)
        n_comp = min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.svd.fit(tfidf)

    def embed_documents(self, texts):
        return self.svd.transform(self.vectorizer.transform(texts)).tolist()

    def embed_query(self, text):
        return self.svd.transform(self.vectorizer.transform([text]))[0].tolist()


def build_contextual_chunks(df, threshold=400):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=25, length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    documents, metadatas = [], []
    for _, row in df.iterrows():
        prefix = f"[Video: {str(row['title'])[:80]}] [Perspective: {row['perspective']}] "
        text = row["light_clean_text"]
        pieces = [text] if len(text) <= threshold else splitter.split_text(text)
        meta = {
            "comment_id": row["comment_id"], "video_id": row["video_id"],
            "title": row["title"], "perspective": row["perspective"],
            "final_sentiment_label": row["final_sentiment_label"],
        }
        for piece in pieces:
            documents.append(prefix + piece)
            metadatas.append(meta)
    return documents, metadatas


@st.cache_resource(show_spinner="Building RAG index (first load only, ~1 min)...")
def load_rag_system(_df):
    # Lexical retriever: reuse Notebook 6's fitted TF-IDF (per full comment)
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    tfidf_row_index = pd.read_csv("tfidf_row_index.csv")

    df = _df.merge(
        tfidf_row_index.reset_index().rename(columns={"index": "tfidf_row"}),
        on="comment_id", how="inner"
    ).sort_values("tfidf_row").reset_index(drop=True)

    # Semantic retriever: contextual chunks + LSA embeddings + FAISS
    documents, metadatas = build_contextual_chunks(df)
    embeddings = FastSemanticEmbeddings(documents, n_components=100)
    vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings, metadatas=metadatas)

    return {
        "df": df,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "vectorstore": vectorstore,
    }


def lexical_retrieval(rag, query, k=5):
    query_vec = rag["tfidf_vectorizer"].transform([query])
    scores = cosine_similarity(query_vec, rag["tfidf_matrix"]).flatten()
    top_idx = np.argsort(scores)[::-1][:k]
    df = rag["df"]
    return [(df.iloc[i]["light_clean_text"], scores[i]) for i in top_idx]


def semantic_retrieval(rag, query, k=5, fetch_k=20, lambda_mult=0.5):
    retriever = rag["vectorstore"].as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    )
    docs = retriever.invoke(query)
    return [(d.page_content, d.metadata) for d in docs]


def hybrid_retrieval(rag, query, k=5, alpha=0.5, fetch_k=20):
    semantic_results = semantic_retrieval(rag, query, k=fetch_k)
    lexical_results = lexical_retrieval(rag, query, k=fetch_k)
    scores = {}
    for rank, (text, _) in enumerate(semantic_results):
        scores[text] = scores.get(text, 0) + alpha * (1 - rank / max(len(semantic_results), 1))
    for rank, (text, _) in enumerate(lexical_results):
        scores[text] = scores.get(text, 0) + (1 - alpha) * (1 - rank / max(len(lexical_results), 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def post_process_retrieval(chunks, max_chars=2000):
    seen, parts, total = set(), [], 0
    for c in chunks:
        if c in seen:
            continue
        seen.add(c)
        if total + len(c) > max_chars:
            break
        parts.append(c)
        total += len(c)
    return "\n---\n".join(parts)


def generate_answer(question, context, groq_api_key):
    from groq import Groq
    client = Groq(api_key=groq_api_key, timeout=25.0, max_retries=1)
    system_prompt = (
        "You answer questions using ONLY the provided context, which comes from a YouTube "
        "comment dataset about de-influencing and anti-consumerism. If the context does not "
        "contain enough information to answer confidently, respond with exactly: "
        "\"I don't know based on the available data.\" Do not guess, speculate, or use outside "
        "knowledge. Keep answers concise and cite specific perspectives/sentiment patterns "
        "from the context where relevant."
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content


# ────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ The E-Commerce Shield")
    page = st.radio(
        "Navigate",
        ["Overview", "Perspectives", "Sentiment", "Brands & Entities", "Keywords", "Ask the Data"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Groq API key (for 'Ask the Data')")
    groq_key_input = st.text_input(
        "Groq API key", value=os.environ.get("GROQ_API_KEY", ""),
        type="password", label_visibility="collapsed"
    )

df = load_core_data()
entities_df = load_entities()
kw_overall, kw_perspective, kw_sentiment = load_keywords()


# ────────────────────────────────────────────────────────────────────────
# Hero header
# ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🛡️ The E-Commerce Shield</h1>
        <p>Discourse intelligence on de-influencing, anti-consumerism, and the internet's pushback against viral marketing.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ────────────────────────────────────────────────────────────────────────
# Overview
# ────────────────────────────────────────────────────────────────────────
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Comments", f"{len(df):,}")
    c2.metric("Videos Analyzed", f"{df['video_id'].nunique()}")
    c3.metric("Consumer Perspectives", f"{df['perspective'].nunique()}")
    c4.metric("Unique Entities Mentioned", f"{entities_df['entity_text'].nunique():,}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overall Sentiment")
        sentiment_counts = df["final_sentiment_label"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        fig = px.pie(
            sentiment_counts, names="Sentiment", values="Count", hole=0.55,
            color="Sentiment", color_discrete_map=SENTIMENT_COLORS
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e5e7eb", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Comments by Perspective")
        persp_counts = df["perspective"].value_counts().reset_index()
        persp_counts.columns = ["Perspective", "Count"]
        fig = px.bar(
            persp_counts, x="Count", y="Perspective", orientation="h",
            color="Perspective", color_discrete_sequence=PERSPECTIVE_COLORS
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e5e7eb", showlegend=False,
                           yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────
# Perspectives
# ────────────────────────────────────────────────────────────────────────
elif page == "Perspectives":
    st.subheader("Sentiment Composition by Perspective")
    cross = pd.crosstab(df["perspective"], df["final_sentiment_label"], normalize="index") * 100
    cross = cross.reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0)
    fig = go.Figure()
    for sentiment in ["Negative", "Neutral", "Positive"]:
        fig.add_trace(go.Bar(
            y=cross.index, x=cross[sentiment], name=sentiment,
            orientation="h", marker_color=SENTIMENT_COLORS[sentiment]
        ))
    fig.update_layout(
        barmode="stack", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb", xaxis_title="Share of Comments (%)", legend_title=""
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explore a Perspective")
    selected = st.selectbox("Choose a perspective", sorted(df["perspective"].unique()))
    subset = df[df["perspective"] == selected]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Comments in this perspective", f"{len(subset):,}")
        top_kw = kw_perspective[kw_perspective["perspective"] == selected].head(10)
        st.markdown("**Top keywords**")
        st.dataframe(top_kw[["term", "tfidf_score"]], hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**Sample comments**")
        sample = subset.sample(min(5, len(subset)), random_state=42)
        for _, row in sample.iterrows():
            st.markdown(
                f"<div class='chat-answer'>"
                f"<span class='source-chip'>{row['final_sentiment_label']}</span>"
                f"{row['light_clean_text'][:280]}</div>",
                unsafe_allow_html=True
            )

# ────────────────────────────────────────────────────────────────────────
# Sentiment
# ────────────────────────────────────────────────────────────────────────
elif page == "Sentiment":
    st.subheader("Sentiment Over Time")
    if "published_at" in df.columns and df["published_at"].notna().any():
        monthly = df.dropna(subset=["published_at"]).copy()
        monthly["month"] = monthly["published_at"].dt.to_period("M").astype(str)
        trend = monthly.groupby(["month", "final_sentiment_label"]).size().reset_index(name="count")
        fig = px.line(
            trend, x="month", y="count", color="final_sentiment_label",
            color_discrete_map=SENTIMENT_COLORS
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e5e7eb", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timestamp data available for a time trend.")

    st.subheader("Top Keywords by Sentiment")
    tabs = st.tabs(["Negative", "Neutral", "Positive"])
    for tab, sentiment in zip(tabs, ["Negative", "Neutral", "Positive"]):
        with tab:
            subset = kw_sentiment[kw_sentiment["sentiment"] == sentiment].head(15)
            if len(subset):
                fig = px.bar(
                    subset.sort_values("tfidf_score"), x="tfidf_score", y="term", orientation="h",
                    color_discrete_sequence=[SENTIMENT_COLORS[sentiment]]
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="#e5e7eb", yaxis_title="", xaxis_title="TF-IDF Score")
                st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────
# Brands & Entities
# ────────────────────────────────────────────────────────────────────────
elif page == "Brands & Entities":
    st.subheader("Entity Type Distribution")
    type_counts = entities_df["entity_label"].value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    fig = px.bar(type_counts, x="Type", y="Count", color="Type",
                 color_discrete_sequence=PERSPECTIVE_COLORS)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e5e7eb", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sentiment Toward Top Brands/Products")
    brand_ents = entities_df[entities_df["entity_label"].isin(["ORG", "PRODUCT"])]
    top_brands = brand_ents["entity_text"].value_counts().head(10).index
    brand_subset = brand_ents[brand_ents["entity_text"].isin(top_brands)]
    cross = pd.crosstab(brand_subset["entity_text"], brand_subset["final_sentiment_label"])
    cross = cross.reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0).reindex(top_brands)
    fig = go.Figure()
    for sentiment in ["Negative", "Neutral", "Positive"]:
        fig.add_trace(go.Bar(y=cross.index, x=cross[sentiment], name=sentiment,
                              orientation="h", marker_color=SENTIMENT_COLORS[sentiment]))
    fig.update_layout(barmode="stack", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e5e7eb", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 20 Mentioned Entities")
    top_entities = entities_df["entity_text"].value_counts().head(20).reset_index()
    top_entities.columns = ["Entity", "Mentions"]
    st.dataframe(top_entities, hide_index=True, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────
# Keywords
# ────────────────────────────────────────────────────────────────────────
elif page == "Keywords":
    st.subheader("Top 25 Keywords Overall")
    fig = px.bar(
        kw_overall.sort_values("tfidf_score"), x="tfidf_score", y="term", orientation="h",
        color_discrete_sequence=["#7c5cff"]
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e5e7eb", height=700, yaxis_title="", xaxis_title="TF-IDF Score")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Keywords by Perspective")
    selected = st.selectbox("Choose a perspective", sorted(kw_perspective["perspective"].unique()))
    subset = kw_perspective[kw_perspective["perspective"] == selected].sort_values("tfidf_score")
    fig = px.bar(subset, x="tfidf_score", y="term", orientation="h",
                 color_discrete_sequence=["#4caf7d"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#e5e7eb", yaxis_title="", xaxis_title="TF-IDF Score")
    st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────
# Ask the Data (RAG)
# ────────────────────────────────────────────────────────────────────────
elif page == "Ask the Data":
    st.subheader("Ask the Data")
    st.caption(
        "Grounded Q&A over the comment dataset. If the answer isn't in the data, "
        "the system will say so instead of guessing."
    )

    if not groq_key_input:
        st.warning("Paste your Groq API key in the sidebar to use this feature.")
    else:
        rag = load_rag_system(df)

        question = st.text_input("Your question", placeholder="e.g. Why do people criticize de-influencing content?")
        col1, col2 = st.columns([1, 3])
        with col1:
            retrieval_mode = st.selectbox("Retrieval strategy", ["Hybrid", "Semantic", "Lexical"])

        if st.button("Ask", type="primary") and question:
            with st.spinner("Retrieving relevant comments..."):
                if retrieval_mode == "Hybrid":
                    results = hybrid_retrieval(rag, question, k=5)
                    texts = [t for t, _ in results]
                elif retrieval_mode == "Semantic":
                    results = semantic_retrieval(rag, question, k=5)
                    texts = [t for t, _ in results]
                else:
                    results = lexical_retrieval(rag, question, k=5)
                    texts = [t for t, _ in results]

                context = post_process_retrieval(texts)

            with st.spinner("Generating grounded answer..."):
                try:
                    answer = generate_answer(question, context, groq_key_input)
                except Exception as e:
                    answer = f"(Generation failed: {e})"

            st.markdown(f"<div class='chat-answer'>{answer}</div>", unsafe_allow_html=True)

            with st.expander("Show retrieved source comments"):
                for t in texts[:5]:
                    st.markdown(f"- {t[:200]}")

st.markdown("---")
st.caption("CSCI370 Project -- The E-Commerce Shield -- De-Influencing & Anti-Consumerism Discourse Intelligence")
