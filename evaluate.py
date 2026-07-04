"""
python evaluate.py # full run — 20 queries x 6 pipelines
python evaluate.py --dry-run # 2 queries per pipeline to verify setup
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest import result

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama

from pipelines import agentic_rag, baseline_rag, contextual_rag, hybrid_rag, multiquery_rag
from pipelines.hybrid_rag import build_bm25_index

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "ragas_test_dataset.json"
RESULTS_DIR = BASE_DIR / "evaluation_results"

#LLM_MODEL = "llama3.1:8b"
LLM_MODEL = "qwen2.5:14b"
#LLM_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CUSTOMER_ID = "CUST-001"

PIPELINES = [
    "baseline_rag",
    "multiquery_rag",
    "contextual_rag",
    "hybrid_rag",
    "agentic_rag",
]


# Reads ragas_test_dataset.json (20 questions). Dry run only loads the first 2 queries for quick testing.
def load_dataset(dry_run: bool) -> list[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:2] if dry_run else data

#  Loads the ChromaDB databases
def setup_vectorstores():
    print("Loading vectorstores...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    kb_vs = Chroma(
        collection_name="knowledge_base",
        persist_directory=str(BASE_DIR / "chroma_db" / "knowledge_base"),
        embedding_function=embeddings,
    )
    persona_vs = Chroma(
        collection_name="customer_personas",
        persist_directory=str(BASE_DIR / "chroma_db" / "customer_personas"),
        embedding_function=embeddings,
    )
    # builds the BM25 index
    bm25, corpus = build_bm25_index(kb_vs)
    print("  Done.\n")
    return kb_vs, persona_vs, bm25, corpus

# Configures RAGAS scoring metrics using llama3.1:8b as the judge
def setup_ragas_metrics():
    print("Configuring RAGAS metrics...")
    llm = LangchainLLMWrapper(ChatOllama(model=LLM_MODEL, temperature=0, format="json"))
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]
    print("  Done.\n")
    return metrics


# Sends the question to whichever pipeline is being evaluated and returns the result.

def run_query(pipeline_name: str, query: str, kb_vs, persona_vs, bm25, corpus) -> dict:
    if pipeline_name == "baseline_rag":
        return baseline_rag.run(query=query, vectorstore=kb_vs)
    elif pipeline_name == "multiquery_rag":
        return multiquery_rag.run(query=query, vectorstore=kb_vs)
    elif pipeline_name == "contextual_rag":
        return contextual_rag.run(query=query, customer_id=CUSTOMER_ID, vectorstore=kb_vs, persona_vectorstore=persona_vs)
    elif pipeline_name == "hybrid_rag":
        return hybrid_rag.run(query=query, vectorstore=kb_vs, bm25=bm25, corpus=corpus)
    elif pipeline_name == "agentic_rag":
        return agentic_rag.run(query=query, vectorstore=kb_vs)

# Pulls the retrieved chunks out of the result dict so RAGAS can score them. Each pipeline stores chunks
# slightly differently, so this handles each case.
def extract_contexts(pipeline_name: str, result: dict) -> list[str]:
    if pipeline_name == "contextual_rag":
        chunks = result.get("persona_chunks", []) + result.get("retrieved_chunks", [])
        return chunks or [result.get("context", "")]
    elif pipeline_name == "agentic_rag":
        raw = result.get("context", "")
        if not raw or raw == "No tool calls made.":
            return ["No context retrieved."]
        return [p.strip() for p in raw.split("\n\n") if p.strip()] or [raw]
    else:
        chunks = result.get("retrieved_chunks", [])
        return chunks or [result.get("context", "")]


#   The main evaluation loop for one pipeline:
#  1. Loops through every test question, runs it through the pipeline, records the answer, contexts, and latency
#  2. Passes all answers to RAGAS for scoring
#  3. Returns the scores and average latency

def evaluate_pipeline(pipeline_name: str, test_cases: list[dict], kb_vs, persona_vs, bm25, corpus, metrics) -> dict:
    print(f"{'='*60}")
    print(f"  {pipeline_name.upper()}")
    print(f"{'='*60}")

    samples = []
    latencies = []

    for i, tc in enumerate(test_cases, 1):
        q = tc["question"]
        print(f"  [{i:02d}/{len(test_cases)}] {q[:65]}{'...' if len(q) > 65 else ''}", end="", flush=True)
        try:
            result = run_query(pipeline_name, q, kb_vs, persona_vs, bm25, corpus)
            answer = result.get("response", "")
            contexts = extract_contexts(pipeline_name, result)
            latency = result.get("latency_seconds", 0.0)
            latencies.append(latency)
            print(f" ({latency:.1f}s)")
        except Exception as exc:
            answer = f"ERROR: {exc}"
            contexts = [""]
            latency = 0.0
            print(f" ERROR: {exc}")

        samples.append({
            "id": tc["id"],
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": tc["ground_truth"],
            "category": tc.get("category", ""),
            "difficulty": tc.get("difficulty", ""),
            "latency_seconds": latency,
        })

    print(f"\n  Scoring with RAGAS...")
    valid = [s for s in samples if not s["answer"].startswith("ERROR:")]
    ragas_samples = [
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s["ground_truth"],
        )
        for s in valid
    ]
    result = evaluate(dataset=EvaluationDataset(samples=ragas_samples), metrics=metrics, run_config=RunConfig(timeout=300, max_workers=1))
    scores = {k: round(float(v), 4) for k, v in result.to_pandas().mean(numeric_only=True).items()}
    for k, v in scores.items():
        print(f"    {k}: {v}")

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "pipeline": pipeline_name,
        "samples": samples,
        "ragas_scores": scores,
        "avg_latency_seconds": round(avg_latency, 3),
    }


# Prints a formatted table to the terminal comparing all 5 pipelines side by side across the 4 metrics plus latency.

def print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 76)
    print("  SUMMARY")
    print("=" * 76)
    headers = ["Pipeline", "Faithfulness", "Ans.Relevancy", "Ctx Precision", "Ctx Recall", "Latency"]
    widths   = [20,         13,             14,              14,              11,           9]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + "-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in all_results:
        s = r["ragas_scores"]
        cells = [
            r["pipeline"],
            str(s.get("faithfulness", "N/A")),
            str(s.get("answer_relevancy", "N/A")),
            str(s.get("context_precision", "N/A")),
            str(s.get("context_recall", "N/A")),
            f"{r['avg_latency_seconds']:.2f}s",
        ]
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(cells, widths)))
    print("=" * 76)

# Saves the results to evaluation_results
def save_results(all_results: list[dict], dry_run: bool) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ragas_{timestamp}{'_dryrun' if dry_run else ''}"

    json_path = RESULTS_DIR / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    csv_path = RESULTS_DIR / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pipeline", "faithfulness", "answer_relevancy", "context_precision", "context_recall", "avg_latency_seconds"])
        for r in all_results:
            s = r["ragas_scores"]
            writer.writerow([
                r["pipeline"],
                s.get("faithfulness", ""),
                s.get("answer_relevancy", ""),
                s.get("context_precision", ""),
                s.get("context_recall", ""),
                r["avg_latency_seconds"],
            ])

    print(f"\n  Saved → {json_path.name}")
    print(f"  Saved → {csv_path.name}")


#  Runs everything in order: 
#  load dataset → load vectorstores → configure metrics → evaluate each pipeline → print summary → save results.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run 2 queries per pipeline instead of 20")
    args = parser.parse_args()

    print("\nShopNest RAG — RAGAS Evaluation")
    if args.dry_run:
        print("(dry-run: 2 queries per pipeline)\n")

    test_cases = load_dataset(args.dry_run)
    print(f"{len(test_cases)} test cases loaded.\n")

    try:
        kb_vs, persona_vs, bm25, corpus = setup_vectorstores()
    except Exception as e:
        print(f"ERROR: Could not load vectorstores: {e}")
        print("Run ingest.py first.")
        sys.exit(1)

    metrics = setup_ragas_metrics()

    all_results = []
    for pipeline_name in PIPELINES:
        result = evaluate_pipeline(pipeline_name, test_cases, kb_vs, persona_vs, bm25, corpus, metrics)
        all_results.append(result)
        print()

    print_summary(all_results)
    save_results(all_results, args.dry_run)


if __name__ == "__main__":
    main()
