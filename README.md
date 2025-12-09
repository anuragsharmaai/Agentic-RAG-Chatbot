# Multi-Agent Research Assistant (LangGraph + Gemini + Pinecone + SerpAPI)

This project provides a Python backend (FastAPI + LangGraph) with two collaborating agents (research and summary) and a Streamlit frontend for quick testing. It integrates Google Gemini for reasoning, SerpAPI for live web search, and Pinecone for a simple RAG store.

## Features
- **Agents**: Research and Summary orchestrated with LangGraph
- **Tools**: Web search (SerpAPI + scraping) and RAG (Pinecone)
- **Safety**: Prompt injection detection, content filter, token limit
- **Apps**: FastAPI backend, Streamlit frontend

## Architecture
- **app/main.py**: FastAPI app and endpoints
- **app/agents/graph.py**: LangGraph with research -> summary flow
- **app/tools/web_search.py**: SerpAPI search + page fetch
- **app/tools/pinecone_tool.py**: Pinecone index, upsert, and search
- **app/safety.py**: Guardrails
- **app/schemas.py**: Pydantic request/response models
- **streamlit_app.py**: Streamlit frontend

## Setup
- **Python**: 3.10+
- **Environment**:
  - `cp .env.example .env`
  - Fill required keys: `GOOGLE_API_KEY`, `SERPAPI_API_KEY`, `PINECONE_API_KEY`
  - Optional: `ALLOWLISTED_DOMAINS`, `SERVER_HOST` (default 0.0.0.0), `SERVER_PORT` (default 8000)
  - Frontend optional env: `API_BASE` (default inside app is `http://localhost:9010`; you can override via Streamlit sidebar)
- **Create & activate a virtual env**:
  - macOS/Linux (bash/zsh):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    ```
  - Windows (PowerShell):
    ```powershell
    py -3 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    ```
- **Install**:
  - `pip install -r requirements.txt`

## Run Backend
- `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- Health: http://localhost:8000/health
- Docs: http://localhost:8000/docs

## Run Frontend
- `streamlit run streamlit_app.py`
- In the sidebar, set **API Base URL** to your backend (e.g., `http://localhost:8000`).
- Alternatively, set env `API_BASE` before launching Streamlit.

## API Endpoints
- **GET /health**
  - Returns `{ "status": "ok" }`

- **POST /api/ingest**
  - Multipart form to ingest a PDF into RAG
  - Fields: `file` (PDF), `metadata` (optional JSON string)
  - Example (curl):
    ```bash
    curl -X POST \
      -F "file=@/path/to/file.pdf" \
      -F 'metadata={"source":"demo"}' \
      http://localhost:8000/api/ingest
    ```

- **POST /api/research**
  - JSON: `{ "query": "What is xyz?", "max_web_results": 5, "max_rag_chunks": 5 }`
  - Returns executive `summary`, `sources`, `web_results`, `rag_passages`

- **POST /api/chat**
  - JSON: `{ "message": "Summarize ...", "conversation_id": "optional" }`
  - Returns `response` and `intermediate_state`

## RAG & Embeddings Notes
- Primary embeddings use SentenceTransformers (dim=768). If unavailable, it falls back to Google embeddings (when `GOOGLE_API_KEY` is set) and finally to a deterministic hash.
- Pinecone index is auto-created if missing. Check `.env` for `PINECONE_INDEX` and region (`PINECONE_ENV`).

## Troubleshooting
- **Empty web results**: Ensure `SERPAPI_API_KEY` is set and domains are allowed via `ALLOWLISTED_DOMAINS`.
- **Pinecone errors**: Verify `PINECONE_API_KEY`, `PINECONE_ENV`, and `PINECONE_INDEX`.
- **Frontend cannot reach backend**: Confirm the backend URL in Streamlit sidebar matches the backend host/port.

## License
MIT
