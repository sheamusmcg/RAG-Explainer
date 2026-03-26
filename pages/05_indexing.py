import streamlit as st
from components.state_manager import has_embeddings, has_index, clear_downstream
from components.rag_utils import build_faiss_index

# ── Prerequisite ─────────────────────────────────────────────────────────
if not has_embeddings():
    st.title("Indexing")
    st.warning("Please generate embeddings first.")
    st.page_link("pages/04_embeddings.py", label="Go to Embeddings", icon=":material/arrow_back:")
    st.stop()

st.title("Indexing")
st.write(
    "Embeddings need to be stored in a structure optimized for fast nearest-neighbor "
    "search. This is what a **vector database** (or vector index) does."
)

embeddings = st.session_state["embeddings"]
st.write(
    f"You have **{embeddings.shape[0]}** vectors of dimension **{embeddings.shape[1]}** "
    "ready to index."
)

if st.button("Build FAISS Index", type="primary"):
    with st.spinner("Building index..."):
        index = build_faiss_index(embeddings)
    clear_downstream("index")
    st.session_state["faiss_index"] = index
    st.success("Index built successfully!")

# ── Results ──────────────────────────────────────────────────────────────
if has_index():
    index = st.session_state["faiss_index"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vectors Indexed", index.ntotal)
    with col2:
        st.metric("Dimensions", index.d)
    with col3:
        st.metric("Index Type", "Flat L2 (exact)")

    st.info(
        "**What just happened?** All embedding vectors were loaded into a FAISS "
        "IndexFlatL2 — an exact-search index that compares your query against every "
        "stored vector using Euclidean (L2) distance. This is simple and accurate for "
        "small datasets. Production systems with millions of vectors use approximate "
        "algorithms (HNSW, IVF) that trade a tiny amount of accuracy for huge speed gains."
    )

    with st.expander("Learn more: Index Algorithms"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**IndexFlatL2 (Exact)**")
            st.write(
                "Brute-force search. Checks every vector. Maximum accuracy but slow "
                "at scale. This is what we're using in the demo."
            )
            st.markdown("**IVF (Approximate)**")
            st.write(
                "Inverted File Index. Clusters vectors and only searches nearby clusters. "
                "Good when speed matters more than perfect recall."
            )
        with col2:
            st.markdown("**HNSW (Approximate)**")
            st.write(
                "Hierarchical Navigable Small World. Graph-based search with excellent "
                "speed-accuracy tradeoff. The most common production choice."
            )
            st.markdown("**PQ (Compressed)**")
            st.write(
                "Product Quantization. Compresses vectors to minimize memory. "
                "Accept minor accuracy loss for significant storage savings."
            )

    st.divider()
    st.page_link(
        "pages/06_retrieval.py",
        label="Next: Retrieve",
        icon=":material/arrow_forward:",
    )
