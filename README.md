# Personalised RAG Chatbot

**Comparing five Retrieval-Augmented Generation (RAG) strategies for personalising an LLM-based e-commerce customer support chatbot.**

MSc Artificial Intelligence dissertation, Heriot-Watt University (supervised by Dr Santiago Chumbe).

[Presentation slides](docs/presentation.pdf)

---

## The problem

LLMs are fluent but ungrounded. RAG fixes factual grounding, but most RAG research focuses on generic retrieval, not personalising support to an individual customer.

This project builds a support chatbot for **ShopNest**, a fictional UK sportswear store, and asks:

1. Which RAG strategy performs best overall for e-commerce support?
2. Does Contextual RAG improve personalisation over generic retrieval?
3. Does Agentic RAG trade latency for completeness?

## Five strategies compared

| Strategy | Retrieval | Personalisation | Trade-off |
|---|---|---|---|
| **Baseline** | Dense vector search | Persona chunks reranked with knowledge base | Simple reference point |
| **Multi-Query** | 3 LLM-paraphrased queries, merged | Persona chunks reranked with knowledge base | Wider recall, more LLM calls |
| **Contextual** | Original query + separate persona lookup | Persona chunks + customer name passed to the LLM | Flagship personalisation strategy |
| **Hybrid** | BM25 + dense search, fused | Persona chunks reranked with knowledge base | Built for exact matches (order ID, product ID) |
| **Agentic** | ReAct agent chooses tools dynamically | `GetCustomerProfile()` tool | Most flexible, slowest |

## Architecture

```
Customer question
      │
      ├──► Knowledge base (ChromaDB): FAQs, policies, product catalogue
      ├──► Customer persona store (ChromaDB): profile, orders, loyalty, support history
      │
      ▼
Shared processing pipeline (same for every strategy, for a fair comparison)
  rerank (cross-encoder) → reverse repack → context compression
      │
      ▼
LLM (Llama 3.1 8B via Ollama) → grounded, personalised answer
```

## Evaluation

- **Automated (RAGAS):** 20 general queries + 8 personalised queries, scored on faithfulness, answer relevancy, context precision and context recall.
- **Human evaluation:** 9 participants rated five blind-labelled systems (A–E) on accuracy, clarity, personalisation and trust (1–10), using a Streamlit app.

## Results

| Finding | Result |
|---|---|
| Best human rating | **Contextual RAG: 8.33/10**, most voted "most personalised" |
| Most preferred overall | **Agentic RAG: 7.81/10**, most voted "most preferred" and "most helpful" |
| Best automated scores (general queries) | **Agentic RAG:** faithfulness **0.933**, context precision **0.958**, but slowest (**5.6s**) |
| Best on personalised queries | **Contextual RAG:** answer relevancy 0.69, faithfulness 0.854 |
| Fastest | **Hybrid RAG: 1.87s**, but lowest faithfulness (0.825) |
| Baseline / Multi-Query / Hybrid | 6.4–7.2/10 from human evaluators |

**Conclusion:** strategies that explicitly bring in customer-specific data, through context injection (Contextual) or dynamic tool lookup (Agentic), are clearly preferred by real users over generic retrieval, even where automated metrics don't show one clear winner.

### Limitations and future work
- The human evaluation sample is small (n = 9).
- Contextual RAG doesn't rerank persona and knowledge-base chunks together like the other four strategies. Fixing this and re-evaluating is future work.
- RAGAS results weren't broken down by query category, so whether Hybrid beats Baseline on product/order queries is untested.

## Tech stack

Python · LangChain · LangGraph · ChromaDB · Hugging Face (`BAAI/bge-base-en-v1.5` embeddings, `ms-marco-MiniLM-L-6-v2` cross-encoder) · Ollama (Llama 3.1 8B) · RAGAS · Streamlit · rank-bm25 · Faker

## Project structure

```
pipelines/                 # The five RAG strategies + shared pipeline + LLM wrapper
Knowledge_base/            # ShopNest FAQs, policies and product catalogue
Customer & Persona/        # Synthetic customer data and generator
chroma_db/                 # Pre-built vector stores (knowledge base + personas)
ingest.py                  # Builds the vector stores
evaluate.py                # RAGAS evaluation across all pipelines
app.py                     # Streamlit chat app used for human evaluation
evaluation_results/        # RAGAS results
```

## How to run

**Requirements:** Python 3.11 and [Ollama](https://ollama.com/) with the Llama 3.1 8B model.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the LLM
ollama pull llama3.1:8b

# 3. (Optional) Rebuild the vector stores. Pre-built stores are included in chroma_db/
python ingest.py

# 4. Start the chat app
streamlit run app.py

# 5. Run the RAGAS evaluation
python evaluate.py --dry-run   # quick check: 2 queries per pipeline
python evaluate.py             # full run
```

## Data

All customer data is **synthetic**, generated with [Faker](https://faker.readthedocs.io/). No real customer information is used. Human evaluation participants used anonymous IDs, and their session logs are not included in this repository.
