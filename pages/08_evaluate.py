import streamlit as st
import numpy as np
import pandas as pd
from components.state_manager import (
    has_assembled_prompt,
    has_generated_response,
)
from components.rag_utils import (
    embed_texts,
    compute_context_relevance,
    compute_faithfulness,
    compute_answer_relevance,
)

# ── Prerequisite ─────────────────────────────────────────────────────────
if not has_assembled_prompt():
    st.title("Evaluate")
    st.warning("Please assemble a prompt first.")
    st.page_link(
        "pages/07_augment_generate.py",
        label="Go to Augment & Generate",
        icon=":material/arrow_back:",
    )
    st.stop()

st.title("Evaluate")
st.write(
    "RAG evaluation covers both **retrieval quality** and **generation quality**. "
    "The **RAG Triad** measures three independent relationships."
)

# ── RAG Triad Explanation ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader(":material/filter_list: Context Relevance")
    st.write("Did the retriever return the right information?")
with col2:
    st.subheader(":material/verified: Faithfulness")
    st.write("Is the answer supported by the retrieved context?")
with col3:
    st.subheader(":material/question_answer: Answer Relevance")
    st.write("Does the final answer address the user's question?")

st.divider()

# ── Compute Scores ───────────────────────────────────────────────────────
results = st.session_state["retrieval_results"]
query_text = st.session_state["query_text"]
query_embedding = st.session_state["query_embedding"]
embeddings = st.session_state["embeddings"]
has_response = has_generated_response()
response_text = st.session_state.get("generated_response", "")

if st.button("Compute Evaluation Scores", type="primary"):
    with st.spinner("Computing scores..."):
        # Context Relevance: query vs each retrieved chunk embedding
        retrieved_indices = [r["chunk_index"] for r in results]
        retrieved_embeddings = embeddings[retrieved_indices]
        ctx_score, ctx_per_chunk = compute_context_relevance(
            query_embedding, retrieved_embeddings
        )

        # Faithfulness & Answer Relevance (only if response exists)
        faith_score = None
        ans_rel_score = None
        if has_response and response_text:
            chunk_texts_list = [r["text"] for r in results]
            faith_score = compute_faithfulness(response_text, chunk_texts_list)

            # Embed the response in the same space
            method = st.session_state.get("embedding_method", "tfidf")
            chunks = st.session_state["chunks"]
            fit_texts = [c["text"] for c in chunks] if method == "tfidf" else None
            response_emb = embed_texts([response_text], method=method, fit_texts=fit_texts)
            ans_rel_score = compute_answer_relevance(query_embedding, response_emb)

    st.session_state["eval_scores"] = {
        "context_relevance": ctx_score,
        "context_per_chunk": ctx_per_chunk,
        "faithfulness": faith_score,
        "answer_relevance": ans_rel_score,
    }
    st.success("Evaluation complete!")

# ── Display Scores ───────────────────────────────────────────────────────
if st.session_state.get("eval_scores"):
    scores = st.session_state["eval_scores"]

    st.header("Scores")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Context Relevance",
            f"{scores['context_relevance']:.1%}",
            help="Average cosine similarity between query and retrieved chunk embeddings.",
        )
    with col2:
        if scores["faithfulness"] is not None:
            st.metric(
                "Faithfulness",
                f"{scores['faithfulness']:.1%}",
                help="Fraction of response sentences grounded in retrieved chunks.",
            )
        else:
            st.metric("Faithfulness", "N/A")
            st.caption("Generate a response to measure faithfulness.")
    with col3:
        if scores["answer_relevance"] is not None:
            st.metric(
                "Answer Relevance",
                f"{scores['answer_relevance']:.1%}",
                help="Cosine similarity between query and response embeddings.",
            )
        else:
            st.metric("Answer Relevance", "N/A")
            st.caption("Generate a response to measure answer relevance.")

    # ── Per-chunk breakdown ──────────────────────────────────────────────
    st.subheader("Per-Chunk Context Relevance")
    chunk_scores_df = pd.DataFrame({
        "Rank": [r["rank"] for r in results],
        "Source": [r["source_title"] for r in results],
        "Relevance": [f"{s:.1%}" for s in scores["context_per_chunk"]],
        "Preview": [r["text"][:100] + "..." for r in results],
    })
    st.dataframe(chunk_scores_df, use_container_width=True, hide_index=True)

    st.info(
        "**Interpreting scores:**\n"
        "- **Context Relevance > 70%** — retriever is finding relevant chunks\n"
        "- **Faithfulness > 80%** — response is well-grounded in context\n"
        "- **Answer Relevance > 60%** — response addresses the question\n\n"
        "Low context relevance? Try different chunk sizes or a different query. "
        "Low faithfulness? The LLM may be hallucinating beyond the provided context."
    )

    # ── Learn more ───────────────────────────────────────────────────────
    with st.expander("Learn more: Evaluation Frameworks"):
        st.write(
            "This demo uses simple heuristic metrics. Production RAG systems use "
            "specialized frameworks:\n\n"
            "- **RAGAS** — Reference-free evaluation using LLM-based judgments. "
            "Measures context precision, context recall, faithfulness, and answer relevancy.\n\n"
            "- **Continuous-Eval** — Modular evaluation that independently assesses each "
            "pipeline component (retriever, reranker, generator).\n\n"
            "- **LangSmith** — Full pipeline tracing with A/B testing for prompt variations "
            "and production monitoring."
        )

    with st.expander("Learn more: RAG vs. Standard LLM Evaluation"):
        st.write(
            "Standard LLM evaluation asks: *Is the output good?*\n\n"
            "RAG evaluation is more nuanced because the pipeline has multiple stages:\n"
            "- **Retrieval metrics** — Did we find the right documents?\n"
            "- **Generation metrics** — Is the answer faithful to those documents?\n"
            "- **End-to-end metrics** — Is the final answer actually helpful?\n\n"
            "You need all three to understand where your system is breaking down."
        )

    # ── Wrap up ──────────────────────────────────────────────────────────
    st.divider()
    st.success(
        "You've completed the full RAG pipeline! Go back to any step to "
        "experiment with different settings — try different chunk sizes, queries, "
        "or documents and see how the scores change."
    )
    st.page_link(
        "pages/01_welcome.py",
        label="Back to Welcome",
        icon=":material/arrow_back:",
    )
