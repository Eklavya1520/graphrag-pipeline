# Developer Notes

Personal notes from building this. Keeping it around because it's useful to remember *why* certain decisions were made.

---

## Why not just use pure vector RAG?

Started with a standard FAISS-based RAG setup. It worked fine for direct factual questions but completely broke down on anything relational. Example:

> *"Which NLP techniques are related to BERT?"*

With vector RAG, you'd get chunks that *mention* BERT, but you'd miss the conceptual graph around it — things like masked language modeling, pre-training objectives, downstream tasks, etc. The graph captures those links naturally.

The tradeoff is ingestion time and complexity. Graph building is slower and needs a good LLM for extraction. For domains where relationships matter (research papers, legal documents, medical knowledge), the investment pays off.

---

## Neo4j schema decisions

Went with a fairly flat schema:
```
(:Entity {name, type, description}) -[:RELATION {type, context}]-> (:Entity)
```

Debated whether to use typed node labels (`:Person`, `:Concept`, `:Technology`) vs a generic `:Entity` with a `type` property. Went with the latter for flexibility — the LLM doesn't always classify entity types consistently, and having everything under `:Entity` means simpler Cypher.

TODO: Add a `:Document` node and connect entities to their source documents. Would make provenance tracking much cleaner.

---

## LangGraph state design

The state dict passed between nodes contains:
- `query` — original user question (never mutated after set)
- `rewritten_query` — Agent B's rewrite (can change each iteration)
- `graph_context` — list of context strings from Neo4j
- `vector_context` — list from FAISS (populated only if graph is thin)
- `answer` — LLM generated answer
- `confidence` — float 0.0–1.0 from critic node
- `iterations` — safety counter, max 3
- `messages` — full conversation history for the LLM

One thing that tripped me up early: LangGraph reducer functions. By default, lists in state get *replaced*, not appended. Had to write a custom `add_messages` reducer for `messages`. The LangGraph docs on this are a bit sparse — ended up looking at the source code.

---

## Agent B query rewriting — what actually works

Simple prompt: *"Rewrite this query to be more specific"* → garbage output.

Better approach — give the LLM explicit intent: extract the main entity, action, and any constraints from the query. Then reformulate. Added a few-shot examples in the prompt which improved consistency a lot.

Also: Agent B now returns entity keywords alongside the rewritten query. These seed the Cypher `WHERE n.name CONTAINS` filters, which is crude but works well enough.

---

## Ragas setup headaches

Ragas >= 0.1 changed its API from the older `0.0.x` versions. Make sure you're using the new `EvaluationDataset` + `evaluate()` interface, not the old `Dataset` approach.

Also Ragas internally calls an LLM (OpenAI by default) to score faithfulness. This means evaluation has a cost. For our 15-question test set, it ran up about $0.06 in API calls — totally fine, just don't accidentally run it on thousands of samples.

---

## Self-correction loop — did it actually help?

Ran an informal ablation: pipeline with vs without the critic + retry loop.

- Without correction: faithfulness ≈ 0.71
- With correction (max 3 retries): faithfulness ≈ 0.82

Mostly helped on questions where the initial graph retrieval pulled in irrelevant neighbouring nodes. The rewritten query on retry 2 was usually much more targeted.

The confidence score is a known hack — it's just asking the LLM to score itself (0.0–1.0). A better approach would be a trained uncertainty estimator, but that's well out of scope for this project.

---

## Things I'd do differently next time

1. **Community detection** — Neo4j's GDS library has Louvain community detection. Running it on the graph and using community membership to filter context would be smarter than pure k-hop traversal.

2. **Async ingestion** — Agent A currently processes documents sequentially. Should be straightforward to parallelise with `asyncio.gather()`. Tested on 20 documents and it takes about 4 minutes sequentially; async would bring this down to under a minute.

3. **Streaming** — LangGraph supports streaming node outputs. Adding this would make the demo feel much more interactive rather than waiting for the full response.

4. **Graph-to-text summarization** — Instead of just dumping raw Cypher results as context, summarize each subgraph component first. Reduces token count and is easier for the LLM to reason over.

5. **Multi-hop traversal** — Currently only doing 1-hop neighbourhood queries. For questions like "What connects BERT to GPT?", you'd need 2–3 hop paths. The `cypher_templates.py` has a `TWO_HOP_PATH_QUERY` that isn't wired up yet.
