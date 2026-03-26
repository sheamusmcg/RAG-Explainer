import streamlit as st
import pandas as pd
from components.state_manager import has_documents, clear_downstream
from components.sample_data import get_sample_corpora

st.title("Load Documents")
st.write(
    "The first step in any RAG pipeline is gathering the documents that will form your "
    "knowledge base. Pick a sample corpus or paste your own text."
)

corpora = get_sample_corpora()

tab_sample, tab_custom = st.tabs(["Sample Corpus", "Paste Custom Text"])

# ── Tab 1: Sample Corpus ─────────────────────────────────────────────────
with tab_sample:
    corpus_name = st.selectbox("Choose a corpus", list(corpora.keys()))
    docs = corpora[corpus_name]

    preview = pd.DataFrame(
        {
            "Title": [d["title"] for d in docs],
            "Words": [len(d["text"].split()) for d in docs],
            "Preview": [d["text"][:120] + "..." for d in docs],
        }
    )
    st.dataframe(preview, use_container_width=True, hide_index=True)

    if st.button("Load Corpus", type="primary", key="load_sample"):
        clear_downstream("documents")
        st.session_state["documents"] = docs
        st.session_state["doc_source"] = "sample"
        st.success(f"Loaded **{len(docs)}** documents from *{corpus_name}*.")

# ── Tab 2: Paste Custom Text ─────────────────────────────────────────────
with tab_custom:
    st.write("Add one or more documents by pasting text below.")

    # Temp list for building up custom docs before committing
    if "_custom_docs" not in st.session_state:
        st.session_state["_custom_docs"] = []

    doc_title = st.text_input("Document title", key="custom_title")
    doc_text = st.text_area(
        "Document text",
        height=200,
        key="custom_text",
        placeholder="Paste your document text here...",
    )

    if st.button("Add Document", key="add_custom"):
        if not doc_title.strip() or not doc_text.strip():
            st.error("Both title and text are required.")
        else:
            st.session_state["_custom_docs"].append(
                {"title": doc_title.strip(), "text": doc_text.strip()}
            )
            st.success(f"Added *{doc_title.strip()}*.")
            st.rerun()

    if st.session_state["_custom_docs"]:
        st.subheader("Documents added so far")
        custom_preview = pd.DataFrame(
            {
                "Title": [d["title"] for d in st.session_state["_custom_docs"]],
                "Words": [len(d["text"].split()) for d in st.session_state["_custom_docs"]],
            }
        )
        st.dataframe(custom_preview, use_container_width=True, hide_index=True)

        if st.button("Use These Documents", type="primary", key="load_custom"):
            clear_downstream("documents")
            st.session_state["documents"] = list(st.session_state["_custom_docs"])
            st.session_state["doc_source"] = "custom"
            st.success(
                f"Loaded **{len(st.session_state['_custom_docs'])}** custom documents."
            )

# ── Current State ────────────────────────────────────────────────────────
st.divider()

if has_documents():
    docs = st.session_state["documents"]
    st.write(f"**{len(docs)} documents loaded** — ready for chunking.")

    with st.expander("View loaded documents"):
        for doc in docs:
            st.subheader(doc["title"])
            st.write(doc["text"])

    st.page_link(
        "pages/03_chunking.py",
        label="Next: Chunk Documents",
        icon=":material/arrow_forward:",
    )
else:
    st.info("No documents loaded yet. Choose a sample corpus or paste your own above.")
