# DocuAgent — Agentic Tool-Use Layer

An agentic loop on top of a RAG pipeline. Instead of only answering from
retrieved document chunks, the agent can **search the web, run Python code,
read/write files, analyze tabular data, and retrieve from ChromaDB** — choosing
which tool to use at each step based on your goal.

## Project layout

```
ASTRA/
├── agent/
│   ├── __init__.py
│   ├── core.py          ← agentic turn loop (Claude + OpenAI transports, SSE streaming)
│   └── tools.py         ← all tool implementations + JSON-schema specs
├── agent_router.py      ← FastAPI router: /agent/run, /agent/stream, /agent/workspace
├── main.py              ← FastAPI entry point  →  uvicorn main:app --reload
├── frontend/
│   └── index.html       ← glassmorphism UI with live transcript panel
├── agent_workspace/     ← sandboxed directory where agent writes/reads files
├── requirements.txt
└── .env                 ← your API keys (not committed to git)
```

## How this maps to Astra's headline features

| Astra capability | This layer's version | Location |
|---|---|---|
| Computer use / browsing | `web_search` tool | `agent/tools.py` |
| Software engineering | `run_python` tool | `agent/tools.py` |
| Office/doc automation | `read_file` / `write_file` tools | `agent/tools.py` |
| Scientific/data analysis | `analyze_dataframe` tool | `agent/tools.py` |
| Document retrieval (RAG) | `search_documents` tool | `agent/tools.py` |
| Long-running multi-step agent | The turn loop itself | `agent/core.py` |
| Live tool-call visibility | SSE streaming endpoint | `agent_router.py` |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
#    Edit .env and replace sk-ant-... with your real Anthropic key
#    (OpenAI key is optional — only needed with provider="openai")

# 3. Run
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** — you'll see the interactive demo UI.

## Try it via curl

**Blocking endpoint** (waits for the full answer):
```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "Calculate sqrt(144) with Python, then write the result to result.txt"}'
```

**Streaming endpoint** (server-sent events):
```bash
curl -N "http://localhost:8000/agent/stream?goal=Search+for+latest+AI+news"
```

**Workspace contents**:
```bash
curl http://localhost:8000/workspace
```

**Health check**:
```bash
curl http://localhost:8000/health
```

## Enabling ChromaDB document search

The `search_documents` tool is included but no-ops gracefully until you wire
up a ChromaDB collection:

```bash
# 1. Uncomment chromadb in requirements.txt, then:
pip install chromadb

# 2. Add to .env:
CHROMA_COLLECTION=docuagent

# 3. Ingest your documents (example using chromadb directly):
python -c "
import chromadb, os
client = chromadb.PersistentClient(path='./chroma_db')
col = client.get_or_create_collection('docuagent')
col.add(documents=['Your document text here'], ids=['doc1'])
print('Ingested.')
"
```

The agent will now call `search_documents` before `web_search` whenever
it thinks the answer might be in your document store.

## Next steps (in order of payoff)

1. **Streaming UI polish** — The SSE stream is live; wire reconnect logic
   and an abort button to the frontend for a more robust UX.
2. **Real search API** — Swap the DuckDuckGo HTML scrape in `web_search`
   for [Tavily](https://tavily.com) or [Serper](https://serper.dev) for
   reliable, production-grade results.
3. **Sandbox `run_python`** — For any public deployment, run the subprocess
   inside a locked-down container (gVisor, Firecracker) rather than bare
   subprocess.
4. **Ingest a document corpus** — Point ChromaDB at your real PDFs/docs and
   the agent can reason across both your knowledge base and the live web.
5. **Autonomous browser control** — A separate project: Playwright + screenshot
   loop + vision model. Worth building after this pattern is solid.
