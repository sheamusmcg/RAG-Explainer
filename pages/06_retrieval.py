import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from components.state_manager import has_index, has_retrieval_results, clear_downstream
from components.rag_utils import (
    embed_texts,
    search_index,
    l2_to_similarity,
    reduce_dimensions,
)

# ── Prerequisite ─────────────────────────────────────────────────────────
if not has_index():
    st.title("Retrieval")
    st.warning("Please build an index first.")
    st.page_link("pages/05_indexing.py", label="Go to Indexing", icon=":material/arrow_back:")
    st.stop()

st.title("Retrieval")
st.write(
    "Now ask a question! The system will embed your query using the same model, "
    "search the index for the closest chunks, and return them ranked by similarity."
)

# ── Query Controls ───────────────────────────────────────────────────────
query = st.text_input(
    "Enter your question",
    placeholder="e.g. What did the Voyager spacecraft discover?",
)
top_k = st.slider("Number of results (top-k)", 1, 10, st.session_state.get("top_k", 3))

if st.button("Search", type="primary"):
    if not query.strip():
        st.error("Please enter a question.")
        st.stop()

    with st.spinner("Embedding query and searching..."):
        method = st.session_state.get("embedding_method", "tfidf")
        chunks = st.session_state["chunks"]
        chunk_texts = [c["text"] for c in chunks]

        # Embed query in the same space as the corpus
        query_emb = embed_texts(
            [query.strip()],
            method=method,
            fit_texts=chunk_texts if method == "tfidf" else None,
        )

        index = st.session_state["faiss_index"]
        distances, indices = search_index(index, query_emb, top_k)
        similarities = l2_to_similarity(distances)

    results = []
    for rank, (idx, sim) in enumerate(zip(indices, similarities)):
        if idx < 0:
            continue  # FAISS may return -1 if k > index size
        results.append({
            "rank": rank + 1,
            "chunk_index": int(idx),
            "similarity": float(sim),
            "source_title": chunks[int(idx)]["source_title"],
            "text": chunks[int(idx)]["text"],
        })

    clear_downstream("retrieval")
    st.session_state["query_text"] = query.strip()
    st.session_state["query_embedding"] = query_emb
    st.session_state["retrieval_results"] = results
    st.session_state["top_k"] = top_k
    st.success(f"Found **{len(results)}** results.")

# ── Results ──────────────────────────────────────────────────────────────
if has_retrieval_results():
    results = st.session_state["retrieval_results"]
    query_text = st.session_state["query_text"]

    st.subheader(f"Results for: *{query_text}*")

    for r in results:
        score_pct = f"{r['similarity']:.1%}"
        with st.container(border=True):
            col1, col2 = st.columns([1, 5])
            with col1:
                st.metric(f"Rank {r['rank']}", score_pct)
            with col2:
                st.markdown(f"**Source:** {r['source_title']}")
                st.write(r["text"])

    # ── Visualization: query in embedding space ──────────────────────────
    st.subheader("Query in Embedding Space")
    st.write("Your query (star) projected alongside all chunks. Lines connect to the retrieved chunks.")

    embeddings = st.session_state["embeddings"]
    query_emb = st.session_state["query_embedding"]
    chunks = st.session_state["chunks"]
    source_labels = [c["source_title"] for c in chunks]
    chunk_texts_preview = [c["text"][:80] + "..." for c in chunks]

    # Combine query + chunk embeddings and project
    all_emb = np.vstack([embeddings, query_emb])
    coords = reduce_dimensions(all_emb, n_components=2, method="pca")

    chunk_coords = coords[:-1]
    query_coord = coords[-1]

    fig = px.scatter(
        x=chunk_coords[:, 0], y=chunk_coords[:, 1],
        color=source_labels,
        hover_name=chunk_texts_preview,
        labels={"x": "PC1", "y": "PC2", "color": "Source"},
        title="Query (star) and Chunk Embeddings",
    )

    # Add query point
    fig.add_trace(go.Scatter(
        x=[query_coord[0]], y=[query_coord[1]],
        mode="markers",
        marker=dict(symbol="star", size=18, color="red", line=dict(width=2, color="black")),
        name="Your Query",
        hovertext=query_text,
    ))

    # Add lines to retrieved chunks
    for r in results:
        idx = r["chunk_index"]
        fig.add_trace(go.Scatter(
            x=[query_coord[0], chunk_coords[idx, 0]],
            y=[query_coord[1], chunk_coords[idx, 1]],
            mode="lines",
            line=dict(color="red", width=1, dash="dot"),
            showlegend=False,
        ))

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ── Learn more ───────────────────────────────────────────────────────
    with st.expander("Learn more: Similarity & Reranking"):
        st.write(
            "**Cosine Similarity** measures the angle between two vectors (most common). "
            "**Euclidean Distance** measures straight-line distance. This demo uses L2 distance "
            "converted to a 0-1 similarity score.\n\n"
            "**Reranking** is a second pass that re-orders results using a more accurate model "
            "(like a cross-encoder). The first retrieval stage casts a wide net; reranking "
            "narrows it to the best matches."
        )

    st.divider()
    st.page_link(
        "pages/07_augment_generate.py",
        label="Next: Augment & Generate",
        icon=":material/arrow_forward:",
    )
