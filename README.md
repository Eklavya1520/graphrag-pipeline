# Agentic GraphRAG Pipeline

> A self-correcting, multi-agent Retrieval-Augmented Generation system that stores and retrieves knowledge through a **Neo4j graph database** — combining structural graph traversal with vector similarity search for more accurate, explainable answers.

---

## Why GraphRAG?

Traditional RAG pipelines chunk documents into pieces and retrieve them by embedding similarity. This works okay for factual lookups, but falls apart when questions require relational reasoning — things like *"What researchers collaborated on transformer architectures?"* or *"Which concepts are prerequisites for understanding attention mechanisms?"*

GraphRAG attacks this by storing knowledge as a **graph of entities and relationships** first, then doing targeted Cypher traversals to gather richer, more structured context before calling the LLM.

**Vector search is still there** — it kicks in as a fallback when graph traversal returns sparse results. The two work together.

---

## Architecture

```
Ingestion (Agent A)
  Raw Text ──► Entity/Relation Extraction (LLM) ──► Neo4j Graph

Query (Agent B + Pipeline)
  User Query
      │
      ▼
  [Query Rewriter — Agent B]
      │  Expands vague queries into structured intent
      ▼
  [Graph Retriever]
      │  Cypher traversal on Neo4j
      ▼
  [Vector Fallback]  ← triggers when graph context is thin
      │
      ▼
  [Answer Generator]
      │
      ▼
  [Self-Correction Critic]
      │  confidence < threshold? → rewrite and retry (max 3x)
      ▼
  Final Answer + Source Trace
      │
      ▼
  [Ragas Evaluator]  ← offline evaluation on test sets
```

### Agent A — Graph Builder
Runs during ingestion. Reads raw `.txt` documents, uses an LLM with structured output to extract `(entity, relation, entity)` triples, then bulk-loads them into Neo4j using `MERGE` to avoid duplicate nodes.

### Agent B — Query Rewriter
Runs at query time. Takes the user's raw question and rewrites it into a cleaner, more specific form. Also extracts entity keywords used to seed the Cypher query.

---

## Project Structure

```
graphrag-pipeline/
├── src/
│   ├── agents/
│   │   ├── graph_builder.py   # Agent A: text → Neo4j
│   │   ├── query_rewriter.py  # Agent B: query rewriting
│   │   └── answer_gen.py      # LLM answer + confidence scoring
│   ├── graph/
│   │   ├── neo4j_client.py    # Neo4j driver wrapper
│   │   ├── cypher_templates.py
│   │   └── graph_retriever.py
│   ├── vector/
│   │   └── vector_store.py    # FAISS/Chroma fallback
│   ├── pipeline/
│   │   ├── state.py           # LangGraph state schema
│   │   ├── nodes.py           # All LangGraph node functions
│   │   ├── edges.py           # Conditional routing logic
│   │   └── graph.py           # Graph assembly + compilation
│   └── evaluation/
│       └── ragas_eval.py      # Ragas metric suite
├── scripts/
│   ├── ingest.py              # Run Agent A ingestion
│   ├── query.py               # Run a query through the pipeline
│   └── evaluate.py            # Run Ragas evaluation suite
├── data/raw/                  # Put your .txt documents here
├── data/eval_questions.json   # Test questions for Ragas
├── notebooks/demo.ipynb       # End-to-end walkthrough
└── tests/
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Neo4j 5.x ([Desktop](https://neo4j.com/download/) or [Aura](https://neo4j.com/cloud/platform/aura-graph-database/) cloud)
- An OpenAI or Google API key

### 2. Clone and install

```bash
git clone https://github.com/yourusername/graphrag-pipeline.git
cd graphrag-pipeline
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and Neo4j credentials
```

### 4. Ingest documents

Drop your `.txt` files into `data/raw/`, then:

```bash
python scripts/ingest.py --data-dir data/raw --build-vector
```

This runs **Agent A** — extracts entities/relationships and loads them into Neo4j. The `--build-vector` flag also builds the FAISS fallback index.

### 5. Run a query

```bash
python scripts/query.py --question "What is the attention mechanism in transformers?"
```

Add `--verbose` to see the retrieved graph context and intermediate state.

### 6. Evaluate

```bash
python scripts/evaluate.py --test-file data/eval_questions.json
```

---

## Ragas Evaluation Metrics

| Metric | Description | Sample Score |
|---|---|---|
| `faithfulness` | Is the answer grounded in retrieved context? | 0.82 |
| `answer_relevancy` | Does the answer address the question? | 0.87 |
| `context_recall` | Was the relevant context retrieved? | 0.79 |
| `context_precision` | Was retrieved context actually useful? | 0.84 |

*Scores above are from a sample run on 15 NLP-domain questions using GPT-4o-mini.*

---

## Self-Correction Loop

The pipeline has a built-in critic that scores answer confidence after each generation:

- **confidence ≥ 0.6** → accept and return
- **confidence < 0.6** → rewrite query differently and retry
- **3 failed attempts** → return best answer with a low-confidence flag

This prevents the model from confidently hallucinating when it doesn't have enough context.

---

## Running Tests

```bash
pytest tests/ -v
```

All tests mock external services (Neo4j + LLM) so they run offline. The test suite covers:
- Entity/relation extraction parsing
- Query rewriter fail-open behaviour
- Edge routing logic (confidence thresholds, iteration limits)
- Individual node outputs


---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| Knowledge Graph | Neo4j 5.x |
| LLM Backend | OpenAI GPT-4o-mini / Google Gemini |
| Vector Fallback | FAISS |
| RAG Evaluation | Ragas |
| Embeddings | OpenAI `text-embedding-3-small` |

---<img width="766" height="632" alt="Screenshot 2026-09-07 230223" src="https://github.com/user-attachments/assets/dba010b1-327f-4d37-8872-6ad923755b95" />
<img width="766" height="632" alt="Screenshot 2026-09-07 230223 - Copy" src="https://github.com/user-attachments/assets/a7e516e3-a5cf-4e81-8383-8634b71963dd" />
<img width="1042" height="872" alt="Screenshot 2026-09-07 230601" src="https://github.com/user-attachments/assets/9fab15b4-2d4a-4098-b3ee-5cee2f00e64e" />
<img width="1018" height="872" alt="Screenshot 2026-09-07 230612" src="https://github.com/user-attachments/assets/b992ad06-1a41-4ce0-ae65-20f80b9b6f03" />
<img width="1035" height="871" alt="Screenshot 2026-09-07 230628" src="https://github.com/user-attachments/assets/5c93ee05-7c93-4534-93f2-9c05552224c5" />
<img width="1027" height="865" alt="Screenshot 2026-09-07 231003" src="https://github.com/user-attachments/assets/1326a91c-6201-413e-8261-03e7e7957314" />
<img width="1031" height="878" alt="Screenshot 2026-09-07 231016" src="https://github.com/user-attachments/assets/6a02e5d7-1de1-46ad-9f19-75c07ba74b40" />




## Known Limitations / Ongoing Work

- Entity extraction quality depends heavily on the LLM — GPT-4o gives noticeably better graph structure than GPT-3.5
- Cypher query generation can fail on very abstract questions (the vector fallback handles these)
- Ragas evaluation requires an LLM internally, so evaluation costs stack up for large test sets
- Currently single-hop graph traversal; multi-hop support is planned (see NOTES.md)

---

## License

MIT
