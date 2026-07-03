# The E-Commerce Shield

Discourse intelligence system for the de-influencing and anti-consumerism trend on YouTube. Scrapes comments, discovers consumer perspectives, scores sentiment, extracts entities and keywords, and answers questions through a RAG system grounded in the actual comment data — with an interactive dashboard tying it all together.

Built for CSCI370, Spring 2026.


## Pipeline Overview

The project runs as 8 sequential notebooks, each consuming the previous notebook's output. Run them in order.

| # | Notebook | Purpose | Key Output |
|---|----------|---------|------------|
| 1 | `Notebook1_DataScraping.ipynb` | Scrapes comments from 12 YouTube videos via YouTube Data API v3 | `comments.csv.gz`, `videos.csv` |
| 2 | `Notebook2_DataCleaning_EDA.ipynb` | 13-step text cleaning pipeline + exploratory analysis | `clean_data.csv.gz` |
| 3 | `Notebook3_Topic_Modeling_PerspectiveMapping.ipynb` | LDA vs. BERTopic, topics mapped to 9 consumer perspectives | `bertopic_with_perspectives.csv` |
| 4 | `Notebook4_SentimentAnalysis.ipynb` | VADER vs. RoBERTa sentiment comparison | `sentiment_results.csv.gz` |
| 5 | `Notebook5_NER.ipynb` | spaCy NER (PERSON, ORG, PRODUCT, MONEY, GPE, DATE) | `ner_comments.csv.gz`, `ner_entities.csv.gz` |
| 6 | `Notebook6_KeywordExtraction.ipynb` | TF-IDF keyword extraction (overall / per perspective / per sentiment) | `keywords_*.csv`, `tfidf_vectorizer.pkl`, `tfidf_matrix.pkl` |
| 7 | `Notebook7_RAG.ipynb` | RAG system: FAISS retrieval (semantic/lexical/hybrid/metadata) + grounded generation | `rag_store/` |
| 8 | `Notebook8_Evaluation.ipynb` | Aggregated evaluation metrics + MLflow logging | `evaluation_summary.csv`, `mlruns/` |

`dashboard_app.py` is a Streamlit app that consumes the outputs of notebooks 2–6 for interactive exploration, plus a live Q&A panel powered by the same retrieval logic as Notebook 7.

## Setup

### Requirements
- Python 3.10+
- A [Groq API key](https://console.groq.com) (free) for the RAG generation step and dashboard Q&A

### Install
```bash
pip install -r requirements.txt
```

### Running the notebooks
Open each notebook in Google Colab or Jupyter and run top to bottom, in order (1 → 8). Each notebook reads the CSV/pkl files produced by the previous one — make sure they're in the same working directory.

Set your Groq API key before running Notebook 7 or 8:
```bash
export GROQ_API_KEY="your-key-here"
```
(In Colab, add it under the 🔑 Secrets panel instead, named `GROQ_API_KEY`.)

### Running the dashboard
Make sure these files (produced by notebooks 2–6) are in the same folder as `dashboard_app.py`:
```
ner_comments.csv.gz
ner_entities.csv.gz
keywords_overall.csv
keywords_by_perspective.csv
keywords_by_sentiment.csv
tfidf_vectorizer.pkl
tfidf_matrix.pkl
tfidf_row_index.csv
```

Then:
```bash
streamlit run dashboard_app.py
```
Paste your Groq API key into the sidebar to use the "Ask the Data" tab.

## Repository Structure
```
.
├── notebooks/
│   ├── Notebook1_DataScraping.ipynb
│   ├── Notebook2_DataCleaning_EDA.ipynb
│   ├── Notebook3_Topic_Modeling_PerspectiveMapping.ipynb
│   ├── Notebook4_SentimentAnalysis.ipynb
│   ├── Notebook5_NER.ipynb
│   ├── Notebook6_KeywordExtraction.ipynb
│   ├── Notebook7_RAG.ipynb
│   └── Notebook8_Evaluation.ipynb
├── dashboard_app.py
├── requirements.txt
├── report/
│   └── CSCI370_Project_Report.docx
└── README.md
```

## Tech Stack
- **Data collection:** YouTube Data API v3
- **NLP:** spaCy, NLTK, scikit-learn, sentence-transformers
- **Topic modeling:** BERTopic, LDA
- **Sentiment:** VADER, RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`)
- **RAG:** FAISS, TF-IDF + Truncated SVD embeddings, Groq (Llama-3.3-70B)
- **Dashboard:** Streamlit, Plotly
- **Monitoring:** MLflow (local tracking)

## Known Limitations
Compute constraints (GPU access lost mid-project) required some documented adaptations — RoBERTa sentiment validated on a sample rather than the full dataset, and RAG embeddings built with TF-IDF+SVD instead of neural sentence-transformers. Full details are in the report's Challenges & Limitations section.

