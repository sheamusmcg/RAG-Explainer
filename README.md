# RAG Explainer

A hands-on, no-code interactive walkthrough of Retrieval-Augmented Generation (RAG) built with Streamlit.

Instead of just reading about RAG, you **do** each step — load documents, chunk them, generate embeddings, build a vector index, run similarity search, assemble a prompt, and evaluate the results.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red) ![License](https://img.shields.io/badge/License-MIT-green)

## The Pipeline

| Step | Page | What You Do |
|------|------|-------------|
| 1 | **Load Documents** | Pick a sample corpus (Space Exploration, Famous Scientists) or paste your own text |
| 2 | **Chunking** | Configure chunk size & overlap, split documents into searchable pieces |
| 3 | **Embeddings** | Convert chunks into numerical vectors, visualize them in 2D (PCA / t-SNE) |
| 4 | **Indexing** | Build a FAISS vector index for fast similarity search |
| 5 | **Retrieval** | Ask a question, find the most relevant chunks with similarity scores |
| 6 | **Augment & Generate** | See how the RAG prompt is assembled; optionally call an LLM |
| 7 | **Evaluate** | Measure Context Relevance, Faithfulness, and Answer Relevance (the RAG Triad) |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501 and follow the steps.

## Features

- **No API keys required** — the core pipeline runs entirely locally using TF-IDF embeddings and FAISS
- **Optional Sentence Transformer** — switch to `all-MiniLM-L6-v2` for semantic embeddings (requires torch)
- **Optional LLM generation** — provide an OpenAI API key to get real responses, or just inspect the assembled prompt
- **Interactive visualizations** — PCA and t-SNE scatter plots of the embedding space with Plotly
- **Step-by-step gating** — each page checks that previous steps are complete before proceeding
- **Educational explanations** — "Learn more" expanders at every step explain the concepts

## Requirements

- Python 3.9+
- streamlit
- pandas, numpy
- scikit-learn
- faiss-cpu
- tiktoken
- plotly
- sentence-transformers (optional, for transformer embeddings)
- openai (optional, for LLM generation)

## Project Structure

```
RAG Explainer/
├── streamlit_app.py          # Entry point and navigation
├── requirements.txt
├── pages/
│   ├── 01_welcome.py         # Overview and key terms
│   ├── 02_load_documents.py  # Load sample or custom documents
│   ├── 03_chunking.py        # Split documents into chunks
│   ├── 04_embeddings.py      # Tokenize and embed chunks
│   ├── 05_indexing.py        # Build FAISS vector index
│   ├── 06_retrieval.py       # Query the index
│   ├── 07_augment_generate.py# Assemble prompt, optional LLM call
│   └── 08_evaluate.py        # RAG Triad evaluation metrics
└── components/
    ├── state_manager.py      # Session state and pipeline gating
    ├── rag_utils.py          # Chunking, embedding, indexing, search, eval
    └── sample_data.py        # Pre-built document corpora
```

## Inspired By

Content adapted from an ODSC AI Engineering Accelerator session on RAG & Grounding.
