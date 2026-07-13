# streamlit run app.py
# Requirements: pip install streamlit

import time
import json
import os
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st

# Page config
st.set_page_config(
    page_title="ShopNest Support",
    layout="centered",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOMERS_JSON = os.path.join(BASE_DIR, "Customer & Persona", "customer_data", "customers.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "session_log.jsonl")

SURVEY_LINK = "https://forms.office.com/Pages/DesignPageV2.aspx?origin=NeoPortalPage&subpage=design&id=8l9CbGVo30Kk245q9jSBPU0_B0gWdLJLiBUgwn0d6IVUQktHTlJQTDZNUENKUlJRRE8zWFpENllSRC4u"

# Mapping of Participant IDs to Customer IDs
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
    "P011": "CUST-011",
    "P012": "CUST-012",
    "P013": "CUST-013",
    "P014": "CUST-014",
    "P015": "CUST-015",
}

# Radio buttons on the sidebar for participant to select
PIPELINE_OPTIONS = [
    "Baseline RAG",
    "Multi-Query RAG",
    "Contextual RAG",
    "Hybrid RAG",
    "Agentic RAG",
]

# hides pipeline identity during human evaluation.
BLIND_LABELS = ["System A", "System B", "System C", "System D", "System E"]

# Logging
def log_interaction(participant_id: str, customer_id: str, pipeline: str, query: str, result: dict):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "participant_id": participant_id,
        "customer_id": customer_id,
        "pipeline": pipeline,
        "query": query,
        "response": result.get("response", ""),
        "latency_seconds": result.get("latency_seconds", 0),
        "enriched_query": result.get("enriched_query"),
        "persona_chunks": result.get("persona_chunks"),
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

# Reads customers.json and caches it into memory to avoid reloading on every message
@st.cache_resource
def load_customers():
    with open(CUSTOMERS_JSON, "r", encoding="utf-8") as f:
        return {c["customer_id"]: c for c in json.load(f)}


# Load knowledge base
@st.cache_resource
def load_vectorstore():
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

# Load customer personas
@st.cache_resource
def load_persona_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name="customer_personas",
        persist_directory=os.path.join(BASE_DIR, "chroma_db", "customer_personas"),
        embedding_function=embeddings,
    )

# Builds the keyword search index and caches it for the session.
@st.cache_resource
def load_bm25_index():
    vs = load_vectorstore()
    from pipelines.hybrid_rag import build_bm25_index
    return build_bm25_index(vs)

# receives the participant's message and routes it to whichever of the 5 pipelines is selected, then returns the response.
def run_pipeline(pipeline_name: str, query: str, customer_id: str) -> dict:
    vs = load_vectorstore()

    if pipeline_name == "Baseline RAG":
        from pipelines.baseline_rag import run
        return run(query=query, vectorstore=vs)

    elif pipeline_name == "Multi-Query RAG":
        from pipelines.multiquery_rag import run
        return run(query=query, vectorstore=vs)

    elif pipeline_name == "Contextual RAG":
        from pipelines.contextual_rag import run
        return run(query=query, customer_id=customer_id, vectorstore=vs, persona_vectorstore=load_persona_vectorstore())

    elif pipeline_name == "Hybrid RAG":
        from pipelines.hybrid_rag import run
        bm25, corpus = load_bm25_index()
        return run(query=query, vectorstore=vs, bm25=bm25, corpus=corpus)

    elif pipeline_name == "Agentic RAG":
        from pipelines.agentic_rag import run
        return run(query=query, vectorstore=vs)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "participant_id" not in st.session_state:
    st.session_state.participant_id = None
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None
if "messages_by_pipeline" not in st.session_state:
    st.session_state.messages_by_pipeline = {p: [] for p in PIPELINE_OPTIONS}
if "selected_pipeline" not in st.session_state:
    st.session_state.selected_pipeline = "Baseline RAG"



# Login page for participants to enter their ID and start the session
def login_page():
    st.title("ShopNest Customer Support")
    st.markdown("#### Dissertation Evaluation — Heriot-Watt University")
    st.divider()

    st.markdown(
        "Welcome! This is a research evaluation of an AI-powered customer support chatbot. "
        "Please enter the **Participant ID** provided by the researcher to begin."
    )

    participant_id = st.text_input(
        "Participant ID",
        max_chars=10,
    ).strip().upper()

    if st.button("Start Session", type="primary", use_container_width=True):
        if participant_id in PARTICIPANT_MAP:
            st.session_state.logged_in = True
            st.session_state.participant_id = participant_id
            st.session_state.customer_id = PARTICIPANT_MAP[participant_id]
            st.session_state.messages_by_pipeline = {p: [] for p in PIPELINE_OPTIONS}
            st.rerun()
        else:
            st.error("Participant ID not recognised. Please check with the researcher.")

    st.divider()
    st.caption("This session is for research purposes only. No personal data is collected.")


# Main chat interface
def chat_page():
    customers = load_customers()
    customer = customers.get(st.session_state.customer_id, {})
    customer_name = customer.get("personal_details", {}).get("first_name", "Customer")
    tier = customer.get("account", {}).get("membership_tier", "")

    # Sidebar for participant info and pipeline selection
    with st.sidebar:
        st.markdown(f"### 👤 {customer_name}")
        st.caption(f"Participant: {st.session_state.participant_id}  |  {tier} Member")
        st.divider()

        st.markdown("**Select System**")
        selected_label = st.radio(
            label="System",
            options=BLIND_LABELS,
            index=PIPELINE_OPTIONS.index(st.session_state.selected_pipeline),
            label_visibility="collapsed",
        )
        selected = PIPELINE_OPTIONS[BLIND_LABELS.index(selected_label)]

        if selected != st.session_state.selected_pipeline:
            st.session_state.selected_pipeline = selected
            st.rerun()

        st.divider()
        st.markdown("**Completed a system?**")
        st.link_button(
            "Start the survey",
            url=SURVEY_LINK,
            use_container_width=True,
            type="primary",
        )

        st.divider()
        if st.button("Log Out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main chat area
    st.title("ShopNest Support")
    current_label = BLIND_LABELS[PIPELINE_OPTIONS.index(st.session_state.selected_pipeline)]
    st.caption(f"Currently using: **{current_label}**")
    st.divider()

    current_messages = st.session_state.messages_by_pipeline[st.session_state.selected_pipeline]

    # Display chat history (only for the currently selected system)
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "latency" in msg:
                st.caption(f"Response time: {msg['latency']}s")

    # Chat input
    if prompt := st.chat_input("Ask a question about your order, returns, delivery..."):
        # Display user message
        current_messages.append({"role": "user", "content": prompt})
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

            log_interaction(
                participant_id=st.session_state.participant_id,
                customer_id=st.session_state.customer_id,
                pipeline=st.session_state.selected_pipeline,
                query=prompt,
                result=result,
            )

            st.markdown(response)
            st.caption(f"Response time: {latency}s")

        current_messages.append({
            "role": "assistant",
            "content": response,
            "latency": latency,
            "pipeline": st.session_state.selected_pipeline,
        })

# Checks if the participant is logged in
if st.session_state.logged_in:
    chat_page()
else:
    login_page()
