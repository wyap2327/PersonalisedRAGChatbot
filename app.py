"""
app.py

ShopNest Customer Support Chatbot — Streamlit Interface
Dissertation evaluation app for comparing 5 RAG strategies.

Run: streamlit run app.py
Requirements: pip install streamlit
"""

import time
import json
import os
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopNest Support",
    page_icon="👟",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOMERS_JSON = os.path.join(BASE_DIR, "Customer & Persona", "customer_data", "customers.json")

# MS Forms survey link — replace with your actual link
SURVEY_LINK = "https://forms.office.com/your-survey-link-here"

# Participant ID → Customer ID mapping
# Each participant is assigned a persona for the evaluation session
PARTICIPANT_MAP = {
    "P001": "CUST-001",
    "P002": "CUST-002",
    "P003": "CUST-003",
    "P004": "CUST-004",
    "P005": "CUST-005",
    "P006": "CUST-006",
    "P007": "CUST-007",
    "P008": "CUST-008",
    "P009": "CUST-009",
    "P010": "CUST-010",
}

PIPELINE_OPTIONS = [
    "Baseline RAG",
    "Multi-Query RAG",
    "Contextual RAG",
    "Hybrid RAG",
    "Agentic RAG",
]

PIPELINE_DESCRIPTIONS = {
    "Baseline RAG":     "Standard retrieval — searches the knowledge base directly.",
    "Multi-Query RAG":  "Generates multiple query variants to improve retrieval coverage.",
    "Contextual RAG":   "Personalises responses using your customer profile.",
    "Hybrid RAG":       "Combines semantic search and keyword search for broader retrieval.",
    "Agentic RAG":      "An AI agent that decides what to search and how to answer.",
}


# ── Load customers ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_customers():
    with open(CUSTOMERS_JSON, "r", encoding="utf-8") as f:
        return {c["customer_id"]: c for c in json.load(f)}


# ── Load pipelines (cached so models only load once) ──────────────────────────
@st.cache_resource
def load_vectorstore():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name="knowledge_base",
        persist_directory=os.path.join(BASE_DIR, "chroma_db", "knowledge_base"),
        embedding_function=embeddings,
    )


@st.cache_resource
def load_bm25_index():
    vs = load_vectorstore()
    from pipelines.hybrid_rag import build_bm25_index
    return build_bm25_index(vs)


def run_pipeline(pipeline_name: str, query: str, customer_id: str) -> dict:
    """Route the query to the correct RAG pipeline."""
    vs = load_vectorstore()

    if pipeline_name == "Baseline RAG":
        from pipelines.baseline_rag import run
        return run(query=query, vectorstore=vs)

    elif pipeline_name == "Multi-Query RAG":
        from pipelines.multiquery_rag import run
        return run(query=query, vectorstore=vs)

    elif pipeline_name == "Contextual RAG":
        from pipelines.contextual_rag import run
        return run(query=query, customer_id=customer_id, vectorstore=vs)

    elif pipeline_name == "Hybrid RAG":
        from pipelines.hybrid_rag import run
        bm25, corpus = load_bm25_index()
        return run(query=query, vectorstore=vs, bm25=bm25, corpus=corpus)

    elif pipeline_name == "Agentic RAG":
        from pipelines.agentic_rag import run
        return run(query=query, vectorstore=vs)


# ── Session state defaults ────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "participant_id" not in st.session_state:
    st.session_state.participant_id = None
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_pipeline" not in st.session_state:
    st.session_state.selected_pipeline = "Baseline RAG"
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def login_page():
    st.title("👟 ShopNest Customer Support")
    st.markdown("#### Dissertation Evaluation — Heriot-Watt University")
    st.divider()

    st.markdown(
        "Welcome! This is a research evaluation of an AI-powered customer support chatbot. "
        "Please enter the **Participant ID** provided by the researcher to begin."
    )

    participant_id = st.text_input(
        "Participant ID",
        placeholder="e.g. P001",
        max_chars=10,
    ).strip().upper()

    if st.button("Start Session", type="primary", use_container_width=True):
        if participant_id in PARTICIPANT_MAP:
            st.session_state.logged_in = True
            st.session_state.participant_id = participant_id
            st.session_state.customer_id = PARTICIPANT_MAP[participant_id]
            st.session_state.messages = []
            st.rerun()
        else:
            st.error("Participant ID not recognised. Please check with the researcher.")

    st.divider()
    st.caption("This session is for research purposes only. No personal data is collected.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def chat_page():
    customers = load_customers()
    customer = customers.get(st.session_state.customer_id, {})
    customer_name = customer.get("personal_details", {}).get("first_name", "Customer")
    tier = customer.get("account", {}).get("membership_tier", "")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👤 {customer_name}")
        st.caption(f"Participant: {st.session_state.participant_id}  |  {tier} Member")
        st.divider()

        st.markdown("**Select RAG System**")
        selected = st.radio(
            label="RAG System",
            options=PIPELINE_OPTIONS,
            index=PIPELINE_OPTIONS.index(st.session_state.selected_pipeline),
            label_visibility="collapsed",
        )

        if selected != st.session_state.selected_pipeline:
            st.session_state.selected_pipeline = selected
            st.session_state.messages = []
            st.rerun()

        st.caption(PIPELINE_DESCRIPTIONS[selected])
        st.divider()

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown("**Completed all 5 systems?**")
        st.link_button(
            "Complete the Survey",
            url=SURVEY_LINK,
            use_container_width=True,
            type="primary",
        )

        st.divider()
        if st.button("Log Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Main chat area ────────────────────────────────────────────────────────
    st.title("👟 ShopNest Support")
    st.caption(f"Currently using: **{st.session_state.selected_pipeline}**")
    st.divider()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "latency" in msg:
                st.caption(f"Response time: {msg['latency']}s")

    # Chat input
    if prompt := st.chat_input("Ask a question about your order, returns, delivery..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_pipeline(
                    pipeline_name=st.session_state.selected_pipeline,
                    query=prompt,
                    customer_id=st.session_state.customer_id,
                )
            response = result.get("response", "Sorry, I could not generate a response.")
            latency = result.get("latency_seconds", 0)

            st.markdown(response)
            st.caption(f"Response time: {latency}s")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "latency": latency,
            "pipeline": st.session_state.selected_pipeline,
        })

        st.session_state.total_queries += 1

        # Prompt survey after 5 questions
        if st.session_state.total_queries > 0 and st.session_state.total_queries % 5 == 0:
            st.info(
                f"You have asked {st.session_state.total_queries} questions. "
                f"Remember to complete the survey after testing all 5 systems."
            )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.logged_in:
    chat_page()
else:
    login_page()
