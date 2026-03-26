import streamlit as st
from components.state_manager import init_state

st.set_page_config(
    page_title="RAG Explainer",
    page_icon=":material/auto_stories:",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

pages = {
    "Getting Started": [
        st.Page("pages/01_welcome.py", title="Welcome", icon=":material/auto_stories:"),
    ],
    "Prepare": [
        st.Page("pages/02_load_documents.py", title="Load Documents", icon=":material/upload_file:"),
        st.Page("pages/03_chunking.py", title="Chunking", icon=":material/content_cut:"),
        st.Page("pages/04_embeddings.py", title="Embeddings", icon=":material/scatter_plot:"),
        st.Page("pages/05_indexing.py", title="Indexing", icon=":material/database:"),
    ],
    "Query": [
        st.Page("pages/06_retrieval.py", title="Retrieval", icon=":material/search:"),
        st.Page("pages/07_augment_generate.py", title="Augment & Generate", icon=":material/smart_toy:"),
    ],
    "Measure": [
        st.Page("pages/08_evaluate.py", title="Evaluate", icon=":material/assessment:"),
    ],
}

page = st.navigation(pages)
page.run()
