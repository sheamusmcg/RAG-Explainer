import streamlit as st

st.title("Welcome to the RAG Explainer")
st.write(
    "*A hands-on, no-code walkthrough of Retrieval-Augmented Generation — "
    "the technique that gives LLMs the ability to look things up before they respond.*"
)

st.write(
    "This app walks you through the complete RAG pipeline step by step. "
    "At each stage you'll actually **do** the work — load documents, split them into "
    "chunks, generate embeddings, build a search index, retrieve relevant context, "
    "assemble a prompt, and evaluate the results — all without writing a single line of code."
)

# ── Pipeline Overview ────────────────────────────────────────────────────
st.header("The RAG Pipeline")
st.write("Here's the journey you'll take:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("### 1. Load")
    st.write("Pick a sample corpus or paste your own documents.")
with col2:
    st.markdown("### 2. Chunk")
    st.write("Split documents into smaller pieces for search.")
with col3:
    st.markdown("### 3. Embed")
    st.write("Convert text into numerical vectors that capture meaning.")
with col4:
    st.markdown("### 4. Index")
    st.write("Store vectors in a database optimized for fast similarity search.")

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.markdown("### 5. Retrieve")
    st.write("Ask a question and find the most relevant chunks.")
with col6:
    st.markdown("### 6. Augment")
    st.write("Assemble a prompt with retrieved context for an LLM.")
with col7:
    st.markdown("### 7. Generate")
    st.write("See (or produce) an LLM response grounded in real sources.")
with col8:
    st.markdown("### 8. Evaluate")
    st.write("Measure retrieval quality, faithfulness, and relevance.")

# ── How RAG Works ────────────────────────────────────────────────────────
st.header("How RAG Works")
st.write(
    "Large Language Models are powerful, but they can't know everything. Their training "
    "data has a cutoff date, they can't access your private documents, and they sometimes "
    "make things up. **RAG** solves this by giving the LLM access to an external knowledge "
    "base so it can retrieve real information before generating an answer."
)

st.info(
    "**The core idea:** Instead of relying only on what the model memorized during training, "
    "RAG retrieves relevant documents at query time and injects them into the prompt."
)

# ── Why RAG? ─────────────────────────────────────────────────────────────
st.header("Why Do We Need RAG?")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader(":material/update: Stale Knowledge")
    st.write(
        "Training data has a cutoff. The model can't know what happened yesterday."
    )
with col2:
    st.subheader(":material/psychology_alt: Hallucination")
    st.write(
        "Without context, LLMs may fabricate plausible-sounding but wrong answers."
    )
with col3:
    st.subheader(":material/memory: Context Limits")
    st.write(
        "Even 200K-token windows can't hold an entire knowledge base."
    )

# ── Key Terms ────────────────────────────────────────────────────────────
st.header("Key Terms")

with st.expander("What is an Embedding?"):
    st.write(
        "An **embedding** is a list of numbers (a vector) that represents the meaning "
        "of a piece of text. Similar meanings produce similar vectors."
    )

with st.expander("What is a Vector Database?"):
    st.write(
        "A **vector database** stores embedding vectors and lets you search for the "
        "most similar ones quickly — like a search engine for meaning, not keywords."
    )

with st.expander("What is a Chunk?"):
    st.write(
        "A **chunk** is a smaller piece of a larger document. Documents are split into "
        "chunks before embedding so each piece can be retrieved individually."
    )

with st.expander("What is Faithfulness?"):
    st.write(
        "**Faithfulness** measures whether an LLM's answer is actually supported by "
        "the retrieved context, rather than hallucinated."
    )

# ── Get Started ──────────────────────────────────────────────────────────
st.divider()
st.page_link(
    "pages/02_load_documents.py",
    label="Get Started: Load Documents",
    icon=":material/arrow_forward:",
)
