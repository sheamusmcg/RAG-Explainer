import streamlit as st
from components.state_manager import (
    has_retrieval_results,
    has_assembled_prompt,
    clear_downstream,
)
from components.rag_utils import assemble_prompt, count_tokens, DEFAULT_SYSTEM_PROMPT

# ── Prerequisite ─────────────────────────────────────────────────────────
if not has_retrieval_results():
    st.title("Augment & Generate")
    st.warning("Please run a retrieval query first.")
    st.page_link("pages/06_retrieval.py", label="Go to Retrieval", icon=":material/arrow_back:")
    st.stop()

st.title("Augment & Generate")
st.write(
    "This is where RAG comes together. The retrieved chunks are injected into a prompt "
    "alongside the user's question, then sent to an LLM for a grounded response."
)

results = st.session_state["retrieval_results"]
query_text = st.session_state["query_text"]

# ── Section 1: Prompt Construction ───────────────────────────────────────
st.header("Step 1 — Assemble the Prompt")

st.write("The prompt has three parts:")

system_prompt = st.text_area(
    "System Prompt (editable)",
    value=DEFAULT_SYSTEM_PROMPT,
    height=100,
)

st.subheader("Retrieved Context")
for r in results:
    st.markdown(f"**Chunk {r['rank']}** (from *{r['source_title']}*, similarity {r['similarity']:.1%})")
    st.text(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))

st.subheader("User Query")
st.info(query_text)

if st.button("Assemble Prompt", type="primary"):
    chunk_texts = [r["text"] for r in results]
    prompt = assemble_prompt(query_text, chunk_texts, system_prompt)
    clear_downstream("generation")
    st.session_state["assembled_prompt"] = prompt
    st.success("Prompt assembled!")

# ── Show assembled prompt ────────────────────────────────────────────────
if has_assembled_prompt():
    prompt = st.session_state["assembled_prompt"]

    st.subheader("Full Assembled Prompt")
    st.code(prompt, language="text")

    token_count = count_tokens([prompt])[0]
    st.metric("Prompt Token Count", token_count)

    st.write(
        "This is exactly what would be sent to an LLM. The model reads the system "
        "instructions, the retrieved context, and the user's question, then generates "
        "a response grounded in the provided documents."
    )

    # ── Section 2: Optional LLM Generation ───────────────────────────────
    st.header("Step 2 — Generate a Response (Optional)")
    st.write(
        "If you have an OpenAI API key, you can send this prompt to an LLM and see "
        "the actual response. **This step is optional** — the main point is seeing "
        "how the prompt is constructed."
    )

    api_key = st.text_input("OpenAI API Key (optional)", type="password")

    if api_key:
        if st.button("Generate Response", type="primary"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                with st.spinner("Generating response..."):
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt.split("USER QUESTION:\n", 1)[-1]
                                if "USER QUESTION:\n" in prompt
                                else query_text},
                        ],
                        max_tokens=512,
                    )
                    answer = response.choices[0].message.content
                st.session_state["generated_response"] = answer
                st.success("Response generated!")
            except Exception as e:
                st.error(f"Error calling OpenAI: {e}")
    else:
        st.info(
            "No API key provided — that's fine! You can still see the full prompt above. "
            "To test generation, paste an OpenAI API key."
        )

    if st.session_state.get("generated_response"):
        st.subheader("LLM Response")
        st.write(st.session_state["generated_response"])

    # ── Learn more ───────────────────────────────────────────────────────
    with st.expander("Learn more: Prompt Engineering for RAG"):
        st.write(
            "The system prompt tells the LLM to answer **only** from the provided context. "
            "This is critical for reducing hallucination.\n\n"
            "**Context window budget:** Everything — system prompt, retrieved chunks, "
            "conversation history, and the query — must fit within the model's token limit "
            "(8K–200K depending on the model). Reranking helps you fit more signal into "
            "fewer tokens by only keeping the most relevant chunks."
        )

    st.divider()
    st.page_link(
        "pages/08_evaluate.py",
        label="Next: Evaluate",
        icon=":material/arrow_forward:",
    )
