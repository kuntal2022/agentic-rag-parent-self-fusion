# Agentic RAG — Parent-Child + Self-Query Fusion

A hybrid Retrieval-Augmented Generation system over a marketing "campaign intelligence"
dataset, combining two LangChain retrieval strategies into one custom retriever, and
wrapping it in a **LangGraph** agent that routes each question to the right execution
path — vector retrieval, a pandas aggregation agent, or both — with an **LLM-as-judge**
self-correction loop.

## Why "fusion"?

| Retriever | What it's good at | What it's bad at |
|---|---|---|
| **Self-Query Retriever** | Turning a natural-language question into a semantic search *plus* a structured metadata filter (e.g. `territory = 'North' AND lift_percent > 10`) | Matches are only as good as the chunk they hit — small chunks lack surrounding context |
| **Parent-Document Retriever** | Embedding small, precise chunks for matching, while returning the full parent document for context | Has no notion of structured filtering — it's pure similarity search |

This project's `SelfQueryParentChildRetriver` class combines them: **Self-Query filters and
matches at the small child-chunk level, then the result is resolved back to the full parent
document** before it ever reaches the LLM. You get precise, filterable retrieval *and* full
context in the answer.

## Architecture

```
<img width="777" height="786" alt="image" src="https://github.com/user-attachments/assets/caaa1ca9-a017-488d-b044-eeb34d8b3d5c" />
```


- **`DecionNode`** — classifies whether a question needs vector retrieval (`Regular`),
  the pandas dataframe agent (`Agent`, for aggregations/comparisons like "highest revenue"),
  or `Both`.
- **`retriver_node`** — runs the fused Self-Query + Parent-Child retriever, then compresses
  results (`EmbeddingsFilter` + `LLMChainExtractor`) so only the relevant text reaches the LLM.
- **`Agent_Execution` / `combine_node`** — spin up a pandas dataframe agent to compute real
  aggregate answers directly from the source data.
- **`LLm_AS_judge`** — grades the generated answer for groundedness and either approves it
  or sends it back to `Generation` with feedback, capped at 3 attempts.

## Repository contents

| File | Purpose |
|---|---|
| [`ParentChildSelfRetriever.ipynb`](ParentChildSelfRetriever.ipynb) | The exploratory notebook — builds the retrievers and the LangGraph pipeline step by step, with detailed markdown commentary above every cell, plus three worked test queries (aggregation-only, retrieval-only, and mixed). |
| [`rag_mix_parentself.py`](rag_mix_parentself.py) | A production-shaped **Streamlit chat app** version of the same pipeline (see below). |
| `requirements.txt` | Pinned dependencies for both the notebook and the app. |

## The Streamlit app (`rag_mix_parentself.py`)

The notebook rebuilds the Chroma vector store from scratch every run, which is fine for
exploration but wasteful (and slow/costly, since it re-embeds everything via the OpenAI API)
for a running app. The Streamlit app changes that:

- **Persistent Chroma** — `Chroma(..., persist_directory="chroma_store/")`. On startup, the app
  checks whether the collection already has vectors; if so, it **skips indexing entirely** and
  reuses the existing store. The source CSV is only needed the very first time.
- **Persistent parent store** — the notebook's `InMemoryStore` for parent documents is swapped
  for `create_kv_docstore(LocalFileStore("parent_store/"))`, a disk-backed key-value store. This
  matters because the *child* vectors and the *parent* documents must stay in sync across restarts
  — persisting only the vector store while keeping parents in memory would leave the retriever
  unable to resolve `doc_id → parent document` after a restart.
- **`st.cache_resource`** wraps every expensive object (embeddings, vector store, retrievers,
  the compiled graph) so Streamlit's rerun-on-every-interaction model doesn't reconstruct them
  on every message.
- **In-memory LangGraph checkpointing** — the graph is compiled with
  `builder.compile(checkpointer=InMemorySaver())`, as requested. Each chat turn runs on its own
  fresh `thread_id`; the visible conversation is rendered from `st.session_state.chat_history`
  instead of relying on the checkpointer to replay it (the `judge_feedback` field uses an
  additive reducer, so reusing one thread across unrelated questions would leak old judge
  feedback into new answers — see the docstring at the top of the script for the full reasoning).
- A debug expander under each answer shows the routing decision, the pandas query used (if any),
  how many judge/regeneration rounds it took, and per-round judge feedback.

## Setup

1. **Clone and install:**

   ```bash
   git clone https://github.com/kuntal2022/agentic-rag-parent-self-fusion.git
   cd agentic-rag-parent-self-fusion
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables** — create a `.env` file (see `.env.example`):

   ```
   OPENAI_API_KEY=sk-...
   ```

   Optional overrides:

   ```
   MAIN_MODEL=gpt-4.1        # orchestrator / generation / self-query LLM
   AGENT_MODEL=gpt-4.1-mini  # pandas dataframe agent + query router
   JUDGE_MODEL=gpt-4.1-mini  # LLM-as-judge
   ```

3. **Provide the source data** — place `campaign_intelligence_sample.csv` at
   `data/campaign_intelligence_sample.csv` (or point the app's sidebar field at any path).
   Each row needs: `campaign_id, territory, quarter, cohort, event_description,
   cross_group_description, reactivation_description, final_insight, combined_description,
   revenue, tg_count, cg_count, lift_percent, response_rate_percent, atbms`.
   This is only needed for the **first** run — once `chroma_store/` and `parent_store/` exist,
   the app no longer touches the CSV.

4. **Run the notebook** (optional, for exploration):

   ```bash
   jupyter notebook ParentChildSelfRetriever.ipynb
   ```

5. **Run the app:**

   ```bash
   streamlit run rag_mix_parentself.py
   ```

## Example questions

- *"Which campaign had the highest revenue, and what were the worst 2 campaigns by response rate?"*
  → routed to the pandas **Agent** path.
- *"Tell me about campaign CMP1033 and why its response rate is so low."*
  → routed to the **Regular** vector-retrieval path.
- *"Compare average revenue between North and South, and tell me about CMP1033 and CMP1003."*
  → routed to **Both**, merging computed numbers with retrieved qualitative context.

## Notes & known limitations

- `MultiQueryRetriever` is built in the notebook but not currently wired into the retrieval
  path — it's available as a drop-in if you want extra recall via query rephrasing.
- The pandas dataframe agent runs `allow_dangerous_code=True`, i.e. it executes LLM-generated
  Python against the dataframe. Treat it the same as any other code-execution agent — don't
  point it at untrusted data sources or expose it on a multi-tenant/public deployment without
  additional sandboxing.
- The LLM-as-judge loop is capped at 3 attempts per question to guarantee the graph terminates.
