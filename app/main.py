import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ResearchRequest, ResearchResponse
from app.tools.pinecone_tool import vector_store
from app.agents.graph import compiled_graph
from app.safety import detect_prompt_injection
from app.config import settings
import uuid
import io
import json
from typing import Optional
from pypdf import PdfReader

app = FastAPI(title="Multi-Agent Research Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/ingest")
async def ingest(
    file: Optional[UploadFile] = File(None),
    metadata: Optional[str] = Form(None),
):
    # Multipart PDF upload with optional metadata
    try:
        content = await file.read()
        reader = PdfReader(io.BytesIO(content))
        pages_text = []
        for p in reader.pages:
            try:
                pages_text.append(p.extract_text() or "")
            except Exception:
                pages_text.append("")
        full_text = "\n\n".join(pages_text)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read PDF file")

    meta_obj = {}
    if metadata:
        try:
            meta_obj = json.loads(metadata)
            if not isinstance(meta_obj, dict):
                meta_obj = {"meta": metadata}
        except Exception:
            meta_obj = {"meta": metadata}

    doc_id = os.path.splitext(file.filename or "")[0] or str(uuid.uuid4())
    docs = [{"id": doc_id, "text": full_text, "metadata": meta_obj}]
    vector_store.upsert_documents(docs)
    return {"status": "ingested", "count": 1, "mode": "pdf", "doc_id": doc_id}

@app.post("/api/research", response_model=ResearchResponse)
async def research(req: ResearchRequest):
    if detect_prompt_injection(req.query):
        raise HTTPException(status_code=400, detail="Prompt appears unsafe. Please rephrase.")
    state = {
        "question": req.query,
        "max_web_results": req.max_web_results,
        "max_rag_chunks": req.max_rag_chunks,
    }
    result = compiled_graph.invoke(state)

    summary = result.get("summary", result.get("draft", ""))
    web_results = result.get("web_results", [])
    rag_passages = result.get("rag_passages", [])
    sources = result.get("sources", [])

    return ResearchResponse(
        summary=summary,
        sources=sources,
        web_results=web_results,
        rag_passages=rag_passages,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
